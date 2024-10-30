import copy
import numpy as np
from numpy.matlib import repmat
from scipy.special import logsumexp

from utils.calculation import _estimate_gaussian_covariances_full
from utils.validation import check_patho_clusters
from base_DEM_MD import BaseDEMMD


def estimate_gaussian_log_proba(x, locations, scales, proportions):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    x : see Notations.md
    locations : see Notations.md
    scales : see Notations.md
    proportions : see Notations.md

    Returns
    -------
    log_tau : see Notations.md

    """
    n_samples, n_features = x.shape
    n_components, _ = locations.shape

    log_tau = np.zeros((n_components, n_samples))

    for i in range(n_components):
        try:
            precisions = np.linalg.inv(scales[i])
            xc = x - repmat(locations[i, :], n_samples, 1)
            xc = np.copy(xc, order="F")
            precisions = np.copy(precisions, order="F")

            log_tau[i, :] = -0.5 * (
                np.sum(np.multiply((xc.dot(precisions)), xc), axis=1)
                + np.sum(np.log(np.linalg.eigvalsh(scales[i])))
                + n_features * np.log(2 * np.pi)
            ) + np.log(proportions[i])

        except np.linalg.LinAlgError:
            print("Error with a covariance matrix")
            print(scales[i])
            break

    return np.copy(log_tau, order="F")


class GaussianDEMMD(BaseDEMMD):
    """
    Implementation of DEM-MD for mixture models with Gaussian continuous distributions.
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
        self.p_discrete = None

    def _initialize_mixture_parameters(self, x_cont, x_discr, tau=None):
        """
        Initialization of the mixture parameters.
        Call a parent method to initialize discrete distribution parameters.

        Parameters
        ----------
        x_cont : see Notations.md
        x_discr: see Notations.md
        """

        n_samples, _ = x_cont.shape
        self.n_components = n_samples

        proportions = np.repeat(1.0 / n_samples, n_samples)
        covariances = self._initialize_covariances(x_cont)

        if self.type_discrete_features is not None:
            p_discrete = self._initialize_pdiscrete(x_discr)
        else:
            p_discrete = []

        self.proportions = np.copy(proportions, order="F")
        self.means = np.copy(x_cont, order="F")
        self.covariances = np.copy(covariances, order="F")
        self.p_discrete = copy.deepcopy(p_discrete)

        self.history.save_variables(
            [
                self.proportions,
                self.means,
                self.covariances,
                self.beta,
                self.n_components,
                self.p_discrete,
            ],
            ["proportions", "means", "covariances", "beta", "n_components", "p_discrete"],
        )

    def _estimate_weighted_log_prob(self, x):
        """
        Estimate the weighted log-probabilities, log P(x | Z) + log proportions.
        The log are not normalised (not - log(sum())).

        Parameters
        ----------
            x : see NOTATIONS.md

        Returns
        -------
            weighted_log_prob : array, shape (n_component,n_samples)
        """

        x_cont, x_discr = self.transform_data(x)

        if self.type_discrete_features is not None:
            return estimate_gaussian_log_proba(
                x_cont, self.means, self.covariances, self.proportions
            ) + self._estimate_discrete_log_prob(x_discr)

        return estimate_gaussian_log_proba(x_cont, self.means, self.covariances, self.proportions)

    def _estimate_cov_matrices(self, x, log_tau):
        """Compute estimates of covariance matrix parameters.

        Parameters
        ----------
        x : see NOTATIONS.md
        log_tau : see NOTATIONS.md

        Returns
        -------
        cov_reg : array-like, shape (n_components, n_features, n_features)
            estimates of covariance matrices with small regularisation to avoid degenerate matrices.
        """
        _, n_features = x.shape

        tau = np.exp(log_tau)
        nk = tau.sum(axis=1) + 10 * np.finfo(tau.dtype).eps

        covariances = _estimate_gaussian_covariances_full(
            tau, x, nk, self.means, reg_covar=self.reg_cov
        )

        cov_reg = np.empty((self.n_components, n_features, n_features))
        q_matrix = self.min_dist * np.eye(n_features)
        cov_reg = (1.0 - self.gamma) * covariances + self.gamma * q_matrix

        return cov_reg

    def _m_step(self, x, log_tau, iteration):
        """M step.

        Parameters
        ----------
        x : see NOTATIONS.md
        log_tau : see Notations.md
        iteration: integer

        Raises
        ------
        ValueError
            If clusters annihilation lead to keep 1 or zero clusters
        """

        #############################################
        # Split of continuously and discretly distributed features
        x_cont, x_discr = self.transform_data(x)
        n_samples, n_features = x_cont.shape

        self.prev_pi = copy.deepcopy(self.proportions)
        tau = copy.deepcopy(np.exp(log_tau))
        nk = tau.sum(axis=1) + 10 * np.finfo(tau.dtype).eps

        ##########################
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
        ##########################

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

            # Delete means corresponding to annihilated clusters
            self.means = self.means[mask, :]
            self.prev_means = self.prev_means[mask, :]
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
            #############################
            # Merge superimposed clusters
            elif (self.n_iter_kconst_ > 200) and patho_clusters:
                log_tau = self._merge_clusters(log_tau, None)
            else:
                self.minimal_iteration_number += 50
        ###########################################

        #############################################
        # Update probabilities of discrete parameters
        if self.type_discrete_features is not None:
            p_discrete = self._m_step_discrete(x_discr, log_tau)
            self.p_discrete = copy.deepcopy(p_discrete)
        ###############################################

        ############################
        # Update covariance matrices
        cov_reg = self._estimate_cov_matrices(x_cont, log_tau)
        self.covariances = copy.deepcopy(cov_reg)

    def fit_predict(self, x):
        """
        Main method to run Dynamical EM algorithms to estimate
        mixture models with Gaussian continuous variables.

        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        labels : array of shape (n_samples)
            Hard assignments of samples x to estimated classes.
        """

        #############################################
        # Split of continuously and discretly distributed features,
        # creation of new_index_features with init=True
        ################################################
        x_cont, x_discr = self.transform_data(x, init=True)
        x_cont = np.copy(x_cont, order="F")

        # Initialization of parameters and first e-step()
        self._initialize_mixture_parameters(x_cont, x_discr)
        log_prob_norm, log_tau = self._e_step(x)
        ll = log_prob_norm
        self.history.save_variables(ll, "log_likelihood")

        tau = copy.deepcopy(np.exp(log_tau))
        nk = tau.sum(axis=1) + 10 * np.finfo(tau.dtype).eps

        self.converged_ = False
        iteration = 1

        #######################################
        # First iteration - estimation of means
        self.prev_means = copy.deepcopy(self.means)  # pickle.loads(pickle.dumps(self.means, -1))
        self.means = tau.dot(x_cont) / nk[:, np.newaxis]
        while True:
            try:
                self._m_step(x, log_tau, iteration)
                self.history.save_variables(
                    [self.proportions, self.means, self.covariances, self.beta, self.p_discrete],
                    ["proportions", "means", "covariances", "beta", "p_discrete"],
                )

            except Exception as e:
                print("Exception in fit-predict loop m-step")
                print(e)
                if str(e).startswith("Invalid value for n_components"):
                    self.means = copy.deepcopy(self.prev_means)
                    self.proportions = copy.deepcopy(self.prev_pi)
                break

            log_prob_norm, log_tau = self._e_step(x)
            ll = log_prob_norm
            tau = np.copy(np.exp(log_tau), order="F")
            nk = tau.sum(axis=1) + 10 * np.finfo(tau.dtype).eps
            self.history.save_variables([ll, log_tau.argmax(axis=0)], ["log_likelihood", "labels"])

            #################################
            # Update means - next iteration
            self.prev_means = copy.deepcopy(self.means)
            self.means = np.dot(tau, x_cont) / nk[:, np.newaxis]

            #########################################
            # Check convergence with Aitken criterion
            if iteration >= 4:
                loglikelihood_future = self.history.log_likelihood[-1]
                loglikelihood_actual = self.history.log_likelihood[-2]
                loglikelihood_prev = self.history.log_likelihood[-3]
                loglikelihood_prev2 = self.history.log_likelihood[-4]
                a_k = (loglikelihood_future - loglikelihood_actual) / (
                    loglikelihood_actual - loglikelihood_prev
                )
                loglikelihood_inf = loglikelihood_actual + (
                    loglikelihood_future - loglikelihood_actual
                ) / (1.0 - a_k)
                a_kprev = (loglikelihood_actual - loglikelihood_prev) / (
                    loglikelihood_prev - loglikelihood_prev2
                )
                loglikelihood_infprev = loglikelihood_prev + (
                    loglikelihood_actual - loglikelihood_prev
                ) / (1.0 - a_kprev)
            else:
                loglikelihood_inf = 10.0
                loglikelihood_infprev = 50.0
                loglikelihood_future = -np.inf

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
        _, log_tau = self._e_step(x)
        self.history.save_variables(log_tau.argmax(axis=0), "final_labels")

        return log_tau.argmax(axis=0)
