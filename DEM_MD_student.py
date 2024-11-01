import copy
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, digamma, logsumexp


from utils.validation import check_patho_clusters
from utils.calculation import _estimate_student_covariances_full
from base_DEM_MD import BaseDEMMD


def estimate_student_log_proba(x, locations, scales, dofs, proportions):
    """
    Estimate the log Student pdf.
    Parameters
    ----------
    x : see Notations.md
    locations : see Notations.md
    scales : see Notations.md
    proportions : see Notations.md
    dofs : see Notations.md

    Returns
    -------
    log_prob : array, shape (n_components, n_samples)
    """

    n_samples, n_features = x.shape
    n_components, _ = locations.shape
    prob = np.empty((n_components, n_samples))
    log_det = np.empty((n_components,))

    for k, (mean, scale) in enumerate(zip(locations, scales)):
        precision = np.linalg.inv(scale)
        xc = x - np.tile(mean, (n_samples, 1))
        prob[k, :] = np.sum(np.multiply((xc.dot(precision)), xc), axis=1)
        log_det[k] = np.sum(np.log(np.linalg.eigvalsh(scale)))

    log_mahala = np.log(1 + prob / dofs.reshape(n_components, 1))

    log_prob = (
        gammaln((dofs + n_features) * 0.5)
        - gammaln(dofs * 0.5)
        - 0.5 * (log_det + n_features * np.log(np.pi * dofs) + (dofs + n_features) * log_mahala.T)
        + np.log(proportions)
    ).T

    return log_prob


class StudentDEMMD(BaseDEMMD):
    """
    Implementation of DEM-MD for mixture models with Student continuous distributions.
    """

    def __init__(
        self,
        eps=1e-6,
        gamma=1e-4,
        type_discrete_features=None,
        index_discrete_features=None,
        is_dummy=False,
        random_state=None,
    ):
        super().__init__(
            eps=eps,
            random_state=random_state,
            is_dummy=is_dummy,
            type_discrete_features=type_discrete_features,
            index_discrete_features=index_discrete_features,
        )
        self.gamma = gamma
        self.minimal_iteration_number = 100
        self.beta = 1.0
        self.n_iter = 0
        self.n_iter_kconst_ = 1
        self.converged_ = False

        self._on_init()

    def _on_init(self):
        self.means = None
        self.prev_means = None
        self.proportions = None
        self.prev_pi = None
        self.covariances = None
        self.dofs = None
        self.p_discrete = None

    def _initialize_mixture_parameters(self, x_cont, x_discr):
        """Initialization of the Student mixture parameters.

        Parameters
        ----------
        x_cont : see Notations.md
        x_discr : see Notations.md
        """

        n_samples, _ = x_cont.shape
        self.n_components = n_samples

        proportions = np.repeat(1.0 / n_samples, n_samples)
        covariances = self._initialize_covariances(x_cont)
        dofs = np.tile(10, self.n_components)

        if self.type_discrete_features is not None:
            p_discrete = self._initialize_pdiscrete(x_discr)
        else:
            p_discrete = []

        self.proportions = np.copy(proportions, order="F")
        self.means = np.copy(x_cont, order="F")
        self.covariances = np.copy(covariances, order="F")
        self.dofs = np.copy(dofs, order="F")
        self.p_discrete = copy.deepcopy(p_discrete)

        self.history.save_variables(
            [
                self.proportions,
                self.means,
                self.covariances,
                self.dofs,
                self.beta,
                self.n_components,
                self.p_discrete,
            ],
            [
                "proportions",
                "means",
                "covariances",
                "dofs",
                "beta",
                "n_components",
                "p_discrete",
            ],
        )

    def _estimate_weighted_log_prob(self, x):
        """
        Estimate the weighted log-probabilities, log P(x | Z, U) + log proportions.
        The log are not normalised (not - log(sum())).

        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        weighted_log_prob : array, shape (n_component,n_samples)
        """

        x_cont, x_discr = self.transform_data(x)

        if self.type_discrete_features is not None:
            return estimate_student_log_proba(
                x_cont, self.means, self.covariances, self.dofs, self.proportions
            ) + self._estimate_discrete_log_prob(x_discr)

        else:
            return estimate_student_log_proba(
                x_cont, self.means, self.covariances, self.dofs, self.proportions
            )

    def _estimate_gamma_u(self, x):
        """Estimate conditional expectations of latent variables U.

        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        gamma_u : see Notations.md
        """

        n_samples, n_features = x.shape
        n_components, _ = self.means.shape
        gamma_u = np.empty((n_components, n_samples))

        for k, (mean, scale, df) in enumerate(zip(self.means, self.covariances, self.dofs)):
            precision = np.linalg.inv(scale)
            xc = x - np.tile(mean, (n_samples, 1))
            gamma_u[k, :] = (df + n_features) / (
                df + np.sum(np.multiply((xc.dot(precision)), xc), axis=1)
            )

        return gamma_u

    def _estimate_log_resp(self, x):
        """
        Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in x with respect to
        the current state of the model.
        Estimate expectations of latent variables U needed.

        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        log_prob : Sum of logarithms of probabilities of each sample in x
            Sum( log p(x))

        log_tau : see Notations.md
        gamma_u : see Notations.md

        """
        x_cont, _ = self.transform_data(x)
        gamma_u = self._estimate_gamma_u(x_cont)

        weighted_log_prob = self._estimate_weighted_log_prob(x)
        log_prob_norm = logsumexp(weighted_log_prob, axis=0)

        return log_prob_norm, weighted_log_prob - log_prob_norm[np.newaxis, :], gamma_u

    def _estimate_params(self, x, log_tau, gamma_u):
        """Compute estimates of covariance matrix parameters.

        Parameters
        ----------
        x: see Notations.md
        log_tau : see Notations.md
        gamma_u : see Notations.md

        Returns
        -------
        cov_reg : array-like, shape (n_components, n_features, n_features)
            estimates of covariance matrices with small regularisation to avoid degenerate matrices.
        """

        _, n_features = x.shape

        tau = np.exp(log_tau)

        covariances = _estimate_student_covariances_full(
            tau, gamma_u, x, self.means, reg_covar=self.reg_cov
        )

        cov_reg = np.empty((self.n_components, n_features, n_features))
        q_matrix = self.min_dist * np.eye(n_features)
        cov_reg = (1.0 - self.gamma) * covariances + self.gamma * q_matrix

        return cov_reg

    def _estimate_dofs(self, cur_dofs, log_tau, gamma_u, x, max_iter=1000):
        """_summary_

        Parameters
        ----------
        cur_dofs : array of shape (n_components)
            _description_
        log_tau : see Notations.md
        gamma_u : see Notations.md
        x : see Notations.md
        max_iter : int, optional
            maximal number of iterations in Brent's method, by default 1000

        Returns
        -------
        new_dofs : array of shape (n_components)
            Newly estimated degrees of freedom by Brent's method.
        """
        _, n_features = x.shape
        n_components, _ = gamma_u.shape

        new_dofs = np.zeros((n_components,))

        # Solve the equation numerically using Brent's method
        # for each component of the mixture
        tau = np.exp(log_tau)
        for k in range(n_components):
            vdim = (cur_dofs[k] + n_features) / 2.0
            tau_loggamma_sum = (tau[k, :] * (np.log(gamma_u[k, :]) - gamma_u[k, :])).sum()

            constant_eq = (
                1.0 + 1.0 * (tau_loggamma_sum / tau[k, :].sum()) + digamma(vdim) - np.log(vdim)
            )

            def function(df):
                return -digamma(df / 2.0) + np.log(df / 2.0) + constant_eq

            if np.sign(function(2)) == np.sign(function(200)) == -1:
                new_dofs[k] = 2.0
            elif np.sign(function(2)) == np.sign(function(200)) == 1:
                new_dofs[k] = 200.0
            else:
                new_dofs[k], obj_root = brentq(
                    function, 2, 200, full_output=True, disp=False, maxiter=max_iter
                )
                if not obj_root.converged:
                    print("No solution found by one-line search")

            if new_dofs[k] <= 0.0 or np.isnan(new_dofs[k]):
                raise ValueError(
                    "[_solve_dof_equation] Error, "
                    + "degree of freedom smaller than zero. \n"
                    + "n_components = "
                    + str(n_components)
                    + ". \n"
                    + "cur_dofs[k] = "
                    + str(cur_dofs[k])
                    + ". \n"
                    + "new_dofs[k] = "
                    + str(new_dofs[k])
                    + ". \n"
                    + "constant[k] = "
                    + str(constant_eq)
                    + ". \n"
                    + "resp_log_gamma_sum[k] = "
                    + str(tau_loggamma_sum)
                    + ". \n"
                    + "resp_sum = "
                    + str(tau[k, :].sum())
                    + ". \n"
                    + "resp = "
                    + str(tau)
                    + ". \n"
                )
        return new_dofs

    def _e_step(self, x):
        """E step

        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        mean(log_prob) : Average of logarithms of probabilities of each sample in data
        log_tau : see Notations.md
        gamma_u : see Notations.md

        """

        log_prob_norm, log_tau, gamma_u = self._estimate_log_resp(x)

        return np.mean(log_prob_norm), log_tau, gamma_u

    def _m_step(self, x, log_tau, gamma_u, iteration):
        """Maximization step.

        Parameters
        ----------
        x : see Notations.md
        log_tau : see Notations.md
        gamma_u : see Notations.md
        iteration : integer
            current iteration

        Raises
        ------
        ValueError
            If clusters annihilation leads to keep 1 or zero clusters
        """
        x_cont, x_discr = self.transform_data(x)
        n_samples, n_features = x_cont.shape

        self.prev_pi = copy.deepcopy(self.proportions)
        tau = np.copy(np.exp(log_tau), order="F")
        nk = tau.sum(axis=1) + 10 * np.finfo(tau.dtype).eps

        ###########################
        # Update proportions
        proportions_em = nk / n_samples
        proportions = np.zeros((self.n_components,))
        e_value = np.sum(np.multiply(self.prev_pi, np.log(self.prev_pi)))
        for k in range(self.n_components):
            proportions[k] = proportions_em[k] + self.beta * self.prev_pi[k] * (
                np.log(self.prev_pi[k]) - e_value
            )
        proportions = proportions / np.sum(proportions)
        self.proportions = copy.deepcopy(proportions)
        ###########################

        ########################
        # Update beta parameter
        eta = np.amin([1.0, 0.5 ** (np.floor(n_features / 2.0 - 1.0))])
        left = (
            np.sum(np.exp(-eta * n_samples * np.abs(self.proportions - self.prev_pi)))
            / self.n_components
        )

        max_proportions_em = np.amax(proportions_em)
        max_pi_old = np.amax(self.prev_pi)

        if e_value == 0.0:
            right = 0.0
        else:
            right = (1.0 - max_proportions_em) / (-max_pi_old * e_value)

        self.beta = 1.0 * np.amin([left, right])
        ########################################

        #####################################
        # Annihilation of too small clusters
        mask = self.proportions >= 1.0 / n_samples
        new_k = np.sum(mask)
        if self.n_components != new_k:
            if new_k <= 1:
                raise ValueError("Invalid value for n_components \n Estimation finished")

            self.n_components = new_k

            new_proportions = self.proportions[mask]
            new_proportions = new_proportions / np.sum(new_proportions)
            self.proportions = copy.deepcopy(new_proportions)

            new_log_tau = log_tau[mask, :]
            new_log_tau = new_log_tau - logsumexp(new_log_tau, axis=0)[np.newaxis, :]
            log_tau = copy.deepcopy(new_log_tau)

            gamma_u = gamma_u[mask, :]

            # Delete means and dofs corresponding to annihilated clusters
            self.means = self.means[mask, :]
            self.prev_means = self.prev_means[mask, :]
            self.dofs = self.dofs[mask]
            self.n_iter_kconst_ = 1
        else:
            self.n_iter_kconst_ += 1
        self.history.save_variables(self.n_components, "n_components")
        ###############################################################

        ##################################################################################
        # Check for a pathological solution corresponding to superimposed clusters
        # Desactivate the proportions regularisation after a certain number of iterations.
        if (iteration >= self.minimal_iteration_number) and (
            (self.history.n_components[iteration - 100] == self.n_components)
        ):
            patho_clusters = check_patho_clusters(self.means, self.covariances)
            if not patho_clusters:
                self.beta = 0.0
            ##############################
            # Merge superimposed clusters
            elif self.n_iter_kconst_ > 200 and patho_clusters:
                log_tau, gamma_u = self._merge_clusters(log_tau, gamma_u)
            else:
                self.minimal_iteration_number += 50
        ###############################################

        #############################################
        # Update probabilities of discrete parameters
        if self.type_discrete_features is not None:
            p_discrete = self._m_step_discrete(x_discr, log_tau)
            self.p_discrete = copy.deepcopy(p_discrete)
        ######################################

        ############################
        # Update covariance matrices
        cov_reg = self._estimate_params(x_cont, log_tau, gamma_u)
        self.covariances = copy.deepcopy(cov_reg)
        #########################################

        ##############
        # Update dofs
        self.dofs = self._estimate_dofs(self.dofs, log_tau, gamma_u, x_cont)

    def fit_predict(self, x):
        """Main method to run Dynamical EM algorithms to estimate
        mixture models with Student continuous variables.

        Parameters
        ----------
        x : array, shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples)
            Hard assignments of samples x to estimated classes.
        """

        #############################################
        # Split of continuous and discrete features,
        # creation of new_index_features with init=True
        x_cont, x_discr = self.transform_data(x, init=True)
        x_cont = np.copy(x_cont, order="F")

        #################################################
        # Initialization of parameters and first e-step()
        self._initialize_mixture_parameters(x_cont, x_discr)
        log_prob_norm, log_tau, gamma_u = self._e_step(x)
        ll = log_prob_norm
        self.history.save_variables(ll, "log_likelihood")

        tau = copy.deepcopy(np.exp(log_tau))

        self.converged_ = False
        iteration = 1

        #######################################
        # First iteration - estimation of means
        self.prev_means = copy.deepcopy(self.means)
        self.means = np.dot((tau * gamma_u), x_cont) / (tau * gamma_u).sum(1)[:, np.newaxis]
        while True:
            try:

                self._m_step(x, log_tau, gamma_u, iteration)
                self.prev_means = copy.deepcopy(self.means)
                self.history.save_variables(
                    [
                        self.proportions,
                        self.means,
                        self.covariances,
                        self.dofs,
                        self.beta,
                        self.p_discrete,
                    ],
                    ["proportions", "means", "covariances", "dofs", "beta", "p_discrete"],
                )

            except Exception as e:
                print(e)
                if str(e).startswith("Invalid value for n_components"):
                    self.means = copy.deepcopy(self.prev_means)
                    self.proportions = copy.deepcopy(self.prev_pi)

                break

            log_prob_norm, log_tau, gamma_u = self._e_step(x)
            ll = log_prob_norm
            tau = np.copy(np.exp(log_tau), order="F")
            self.history.save_variables([ll, log_tau.argmax(axis=0)], ["log_likelihood", "labels"])

            #################################
            # Update means - next iteration
            self.prev_means = copy.deepcopy(self.means)
            self.means = np.dot((tau * gamma_u), x_cont) / (tau * gamma_u).sum(1)[:, np.newaxis]

            #########################################
            # Check convergence with Aitken criterion
            if iteration >= 4:
                small_list_ll = [
                    self.history.log_likelihood[-i] for i in reversed(range(1, 5))
                ]  # ordered as [ll[-4],ll[-3],ll[-2],ll[-1]]

                a_k = (small_list_ll[3] - small_list_ll[2]) / (small_list_ll[2] - small_list_ll[1])
                loglikelihood_inf = small_list_ll[2] + (small_list_ll[3] - small_list_ll[2]) / (
                    1.0 - a_k
                )
                a_kprev = (small_list_ll[2] - small_list_ll[1]) / (
                    small_list_ll[1] - small_list_ll[0]
                )
                loglikelihood_infprev = small_list_ll[1] + (small_list_ll[2] - small_list_ll[1]) / (
                    1.0 - a_kprev
                )
            else:
                loglikelihood_inf = 10.0
                loglikelihood_infprev = 50.0

            if self.n_iter_kconst_ >= 4:
                if np.abs(loglikelihood_inf - loglikelihood_infprev) < self.eps:
                    patho_clusters = check_patho_clusters(self.means, self.covariances)
                    if not patho_clusters:
                        self.converged_ = True
                        break

                    iteration += 1
                else:
                    iteration += 1
            else:
                iteration += 1

        self.n_iter = iteration
        self.ll = ll

        # Always do a final e-step to guarantee
        # that the labels returned by fit_predict(x)
        # are always consistent with fit(x).predict(x)
        _, log_tau, _ = self._e_step(x)
        self.history.save_variables(log_tau.argmax(axis=0), "final_labels")

        return log_tau.argmax(axis=0)

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means.shape

        cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        mean_params = n_features * self.n_components
        dofs_params = self.n_components

        nparams_discrete = 0
        if self.type_discrete_features is not None:
            for type_var, p_discrete in zip(self.type_discrete_features, self.p_discrete):
                if type_var == "Multinomial":
                    nparams_discrete += self.n_components * (p_discrete.shape[1] - 1)
                else:
                    nparams_discrete += self.n_components

        return int(
            cov_params + mean_params + dofs_params + self.n_components - 1 + nparams_discrete
        )
