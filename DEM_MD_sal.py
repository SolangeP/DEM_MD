import copy
import numpy as np
from scipy.special import logsumexp, kve
from utils.validation import check_patho_clusters, check_random_state
from base_DEM_MD import BaseDEMMD


def estimate_sal_log_proba(x, locations, scales, alphas, proportions):
    """
    Estimate the log SAL probability.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
    locations : array-like, shape (n_components, n_features)
    scales: shape of (n_components,n_features, n_features)
    alphas : array-like, shape (n_components,n_features)
    proportions : vector-like, shape (n_components,)

    Returns
    -------
    log_prob : array, shape (n_components, n_samples)
    """
    n_samples, n_features = x.shape
    n_components, _ = locations.shape

    log_prob = np.empty((n_components, n_samples))

    nu = (2.0 - n_features) / 2.0

    for k, (mu, scale, alph, prop_k) in enumerate(zip(locations, scales, alphas, proportions)):
        precision = np.linalg.inv(scale)
        alph_mat = alph.reshape(-1, 1)
        log_det = np.log(np.linalg.det(scale))

        if np.isnan(log_det):
            print(f"In estimated_SAL_log_proba of cluster {k} :")
            print("log det = nan, cluster =", k)
            print("scale", scale)
            print("det", np.linalg.det(scale))

        xc = x - mu
        asa = (alph_mat.T.dot(precision)).dot(alph_mat).reshape(-1)
        xsa = (xc.dot(precision)).dot(alph_mat).reshape(-1)
        log_asa = np.log(2 + asa)

        #### Calculate Mahalanobis distance and log
        mahala = np.sum((xc.dot(precision) * xc), axis=1)
        log_maha = np.log(mahala)
        log_maha[log_maha == -np.inf] = 0.0
        u = np.exp(0.5 * (log_asa + log_maha))
        log_bessel = np.log(kve(nu, u)) - u

        t1 = np.log(2) - 0.5 * (log_det + n_features * np.log(2 * np.pi))
        t2 = xsa
        if nu != 0.0:
            t3 = 0.5 * nu * (log_maha - log_asa)
        else:
            t3 = 0.0

        log_prob[k, :] = t1 + t2 + t3 + log_bessel + np.log(prop_k)

    return log_prob


def estimate_locations(log_tau, e1, e2, x):
    """Estimation of locations parameters for Shifted Asymmetric Laplace distributions in a mixture model.

    Parameters
    ----------
    log_tau : see Notations.md
    e1 : see Notations.md
    e2 : see Notations.md
    x : see Notations.md

    Returns
    -------
    locations : see Notations.md
    """

    _, n_features = x.shape
    tau = np.exp(log_tau)
    n_components = tau.shape[0]
    locations = np.empty((n_components, n_features))

    nk = tau.sum(axis=1) + 10 * np.finfo(tau.dtype).eps
    tau_x = np.dot(tau, x)

    for k in range(n_components):
        tau_e1 = np.sum(tau[k] * e1[k])
        prod_tau_e2 = tau[k] * e2[k]
        tau_e2_x = prod_tau_e2.dot(x)
        tau_e2 = np.sum(tau[k] * e2[k])
        locations[k] = (tau_e1 * tau_e2_x - nk[k] * tau_x[k]) / (tau_e1 * tau_e2 - nk[k] ** 2)

    return locations


def estimate_alpha(log_tau, e1, e2, x):
    """
    Estimation of alpha parameters for Shifted Asymmetric Laplace distributions in a mixture model.

    Parameters
    ----------
    log_tau : see Notations.md
    e1 : see Notations.md
    e2 : see Notations.md
    x : see Notations.md

    Returns
    --------
    alphas : see Notations.md
    """

    _, n_features = x.shape
    n_components = log_tau.shape[0]

    alphas = np.empty((n_components, n_features))

    tau = np.exp(log_tau)
    nk = tau.sum(axis=1) + 10 * np.finfo(tau.dtype).eps
    tau_x = np.dot(tau, x)  # (K,2)

    for k in range(n_components):
        tau_e1 = np.sum(tau[k] * e1[k])
        tau_e2 = np.sum(tau[k] * e2[k])
        prod_tau_e2 = tau[k] * e2[k]
        tau_e2_x = prod_tau_e2.dot(x)
        num = tau_e2 * tau_x[k] - nk[k] * tau_e2_x
        denom = tau_e1 * tau_e2 - nk[k] ** 2
        alphas[k] = num / denom

    return alphas


def estimate_scales(log_tau, e1, e2, locations, alphas, x_cont):
    """Estimation of scales parameters for Shifted Asymmetric Laplace distributions in a mixture model.

    Parameters
    ----------
    log_tau : see Notations.md
    e1 : see Notations.md
    e2 : see Notations.md
    locations : see Notations.md
    alphas : see Notations.md
    x_cont : see Notations.md

    Returns
    -------
    scales: see Notations.md
    """
    n_components, n_features_cont = locations.shape

    scales = np.empty((n_components, n_features_cont, n_features_cont))
    tau = np.exp(log_tau)

    for k, (mu, alph) in enumerate(zip(locations, alphas)):
        tau_k = tau[k, :]
        n_cluster = np.sum(tau_k)
        xc = x_cont - mu
        r = np.dot(tau_k, xc) / n_cluster
        r = r.reshape(-1, 1)
        s_mat = np.dot(tau_k * e2[k, :] * xc.T, xc) / n_cluster
        alph_mat = alph.reshape(-1, 1)
        tau_e1 = tau_k.dot(e1[k, :]) / n_cluster

        scales[k] = (
            s_mat - alph_mat.dot(r.T) - r.dot(alph_mat.T) + alph_mat.dot(alph_mat.T) * tau_e1
        )

    return scales


def scheme_temperature(t, b, rb):
    """Temperature scheme used for annealing on posterior probabilities during E step of estimation process.

    Parameters
    ----------
    t : integer
        current iteration
    b : float, scheme parameter
    rb : float, scheme parameter

    Returns
    -------
    float
        the temperature value applied during E step
    """
    return 1.0 + b * np.sin(t / rb) / (t / rb)


class SALDEMMD(BaseDEMMD):
    """
    Implementation of DEM-MD for mixture models with SAL continuous distribution.
    """

    def __init__(
        self,
        eps=1e-4,
        gamma=0.0,
        random_state=None,
        type_discrete_features=None,
        index_discrete_features=None,
        is_dummy=False,
        use_temperature=True,
        temp_b=1.0,
        temp_rb=3.0,
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
        self.use_temp = use_temperature
        self.temp_rb = temp_rb
        self.temp_b = temp_b
        self.n_iter = 0
        self.n_iter_kconst_ = 1
        self.iteration_temp = 1
        self.beta = 1.0
        self.converged_ = False
        self._on_init()

    def _on_init(self):
        self.means = None
        self.prev_means = None
        self.proportions = None
        self.prev_proportions = None
        self.covariances = None
        self.alphas = None
        self.p_discrete = None
        self.temperature = None

    def _initialize_mixture_parameters(self, x_cont, x_discr):
        """Initialization of the mixture parameters.
        Call a parent method to initialize discrete distribution parameters.

        Parameters
        ----------
        x_cont : see Notations.md
        x_discr : see Notations.md
        """

        n_samples, n_features = x_cont.shape
        self.n_components = n_samples

        means = np.copy(x_cont) + self.random_state.normal(0, np.std(x_cont) * 10e-2, x_cont.shape)
        proportions = np.repeat(1.0 / n_samples, n_samples)
        covariances = self._initialize_covariances(x_cont)
        alphas = np.zeros((self.n_components, n_features))
        self.nu = (2.0 - n_features) / 2.0

        if self.type_discrete_features is not None:
            p_discrete = self._initialize_pdiscrete(x_discr)
        else:
            p_discrete = []

        self.proportions = np.copy(proportions, order="F")
        self.means = np.copy(means, order="F")
        self.covariances = np.copy(covariances, order="F")
        self.alphas = np.copy(alphas, order="F")
        self.p_discrete = copy.deepcopy(p_discrete)

        self.history.save_variables(
            [
                self.proportions,
                self.means,
                self.covariances,
                self.alphas,
                self.beta,
                self.n_components,
                self.p_discrete,
            ],
            ["proportions", "means", "covariances", "alphas", "beta", "n_components", "p_discrete"],
        )

    def _estimate_weighted_log_prob(self, x):
        """
        Estimate the weighted log-probabilities, log P(x | Z, E1, E2) + log proportions.
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
            return estimate_sal_log_proba(
                x_cont, self.means, self.covariances, self.alphas, self.proportions
            ) + self._estimate_discrete_log_prob(x_discr)
        # else:
        return estimate_sal_log_proba(
            x_cont, self.means, self.covariances, self.alphas, self.proportions
        )

    def _estimate_e1_e2(self, x):
        """Estimate  expected values of latent variables W and W^-1.

        Parameters
        ----------
        x : see Notations.md

        Returns
        ----
        e1 : see Notations.md
        e2 : see Notations.md

        """

        n_samples, _ = x.shape
        n_components, _ = self.means.shape

        a = np.empty((n_components,))
        b = np.empty((n_components, n_samples))

        e1 = np.empty((n_components, n_samples))
        e2 = np.empty((n_components, n_samples))

        for k, (mu, scale, alph) in enumerate(zip(self.means, self.covariances, self.alphas)):
            alph_mat = alph.reshape(-1, 1)
            precision = np.linalg.inv(scale)
            xc = x - mu
            a[k] = 2 + (alph_mat.T.dot(precision)).dot(alph_mat)
            b[k, :] = np.sum(np.multiply((xc.dot(precision)), xc), axis=1)

            log_b = np.log(b[k, :])
            log_a = np.log(a[k])
            log_b[log_b == -np.inf] = 0.0

            t1 = np.exp(0.5 * (log_b + log_a))  # (n,)
            kv_haut = np.log(kve(self.nu + 1, t1)) - t1  # (n,)
            kv_bas = np.log(kve(self.nu, t1)) - t1  # (n,)

            e1[k, :] = np.exp(0.5 * (log_b - log_a) + (kv_haut - kv_bas))
            e2[k, :] = np.exp(0.5 * (log_a - log_b) + (kv_haut - kv_bas))

            if self.nu != 0:
                e2[k, :] += -np.sign(self.nu) * np.exp(np.log(2) + np.log(np.abs(self.nu)) - log_b)
            e2[k, np.isnan(e2[k, :])] = np.inf
        return e1, e2

    def _estimate_log_tau(self, x):  # FIX why this name ?
        """
        Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in x with respect to
        the current state of the model.

        Parameters
        ----------
        x : see NOTATIONS.md
        Returns
        -------
        log_prob_norm : Sum of logarithms of probabilities of each sample in x
            Sum( log p(x))
        log_tau : see Notations.md
        e1 : see Notations.md
        e2 : see Notations.md
        """

        x_cont, _ = self.transform_data(x)
        e1, e2 = self._estimate_e1_e2(x_cont)

        if self.use_temp:
            self.temperature = scheme_temperature(
                self.iteration_temp, b=self.temp_b, rb=self.temp_rb
            )
            weighted_log_prob = self._estimate_weighted_log_prob(x) * (1.0 / self.temperature)
        else:
            weighted_log_prob = self._estimate_weighted_log_prob(x)

        log_prob_norm = logsumexp(weighted_log_prob, axis=0)

        return log_prob_norm, weighted_log_prob - log_prob_norm[np.newaxis, :], e1, e2

    def _estimate_cov_matrices(self, x, log_tau, e1, e2, cov_comput="original"):
        """_summary_

        Parameters
        ----------
        x : see NOTATIONS.md
        log_tau : see NOTATIONS.md
        e1 : see Notations.md
        e2 : see Notations.md
        cov_comput : str, optional
            Method used for covariance computation. "Fraley" or "Baudry" regularisations availables in this class, by default "original"

        Returns
        -------
        cov_reg : array-like, shape (n_components, n_features, n_features)
            estimates of covariance matrices with small regularisation to avoid degenerate matrices.
        """

        _, n_features = x.shape

        cov_reg = np.empty((self.n_components, n_features, n_features))

        if cov_comput == "original":
            covariances = estimate_scales(log_tau, e1, e2, self.means, self.alphas, x)
            q_matrix = self.min_dist * np.eye(n_features)
            cov_reg = (1.0 - self.gamma) * covariances + self.gamma * q_matrix

        elif cov_comput == "Fraley":
            emp_mean = np.mean(x, axis=0)
            diff_emp = x - emp_mean
            emp_cov = np.dot(diff_emp.T, diff_emp) / len(x)
            w_const = n_features + 2
            w_mat = emp_cov / (self.n_components ** (2 / n_features))
            cov_reg = self._estimate_scales_reg(log_tau, e1, e2, x, w_mat, w_const)

        elif cov_comput == "Baudry":
            emp_mean = np.mean(x, axis=0)
            diff_emp = x - emp_mean
            emp_cov = np.dot(diff_emp.T, diff_emp) / len(x)
            w_const = n_features + 2
            a0 = 0.001
            w_mat = (
                (a0) ** (1.0 / n_features) * emp_cov / np.linalg.det(emp_cov) ** (1.0 / n_features)
            )
            cov_reg = self._estimate_scales_reg(log_tau, e1, e2, x, w_mat, w_const)
        return cov_reg

    def _estimate_scales_reg(self, log_tau, e1, e2, x, w_mat=None, w_const=None):
        """_summary_

        Parameters
        ----------
        log_tau : see NOTATIONS.md
        e1 : see NOTATIONS.md
        e2 : see NOTATIONS.md
        x : see NOTATIONS.md
        w_mat : array-like, optional
            Regularisation parameter, by default None
        w_const : integer, optional
            Regularisation parameter, by default None

        Returns
        -------
        new_scales
            estimated scale matrices
        """
        _, n_features = x.shape
        tau = np.exp(log_tau)

        scales = np.empty((self.n_components, n_features, n_features))
        new_scales = np.empty((self.n_components, n_features, n_features))

        for k, (mu, alph) in enumerate(zip(self.means, self.alphas)):
            tau_k = tau[k, :]
            n_cluster = np.sum(tau_k)
            xc = x - mu
            r = np.dot(tau_k, xc)
            r = r.reshape(-1, 1)
            s_mat = np.dot(tau_k * e2[k, :] * xc.T, xc)
            alph_mat = alph.reshape(-1, 1)
            tau_e1 = tau_k.dot(e1[k, :])

            scales[k] = (
                s_mat - alph_mat.dot(r.T) - r.dot(alph_mat.T) + alph_mat.dot(alph_mat.T) * tau_e1
            )
            new_scales[k] = copy.deepcopy(scales[k])
            new_scales[k] = (new_scales[k] + w_mat) / (n_cluster + w_const + n_features + 1)

        return new_scales

    def _e_step(self, x):
        """E step

        Parameters
        ----------
        x : see NOTATIONS.md

        Returns
        -------
        mean(log_prob) : Average of logarithms of probabilities of each sample in data
        log_tau : see Notations.md
        """

        log_prob_norm, log_tau, e1, e2 = self._estimate_log_tau(x)

        return np.mean(log_prob_norm), log_tau, e1, e2

    def _m_step(self, x, log_tau, e1, e2, iteration, cov_comput):
        """M step.

        Parameters
        ----------
        x : see NOTATIONS.md
        log_tau : see Notations.md
        e1 : see Notations.md
        e2 : see Notations.md
        iteration : int
            current iteration
        cov_comput : str
           method used for covariance computation: "original", "Fraley" or "Baudry" regularisations availables
        ------
        ValueError
            If clusters annihilation leads to keep 1 or zero clusters
        """

        #############################################
        # Split of continuously and discretly distributed features
        x_cont, x_discr = self.transform_data(x)
        n_samples, n_features = x_cont.shape

        self.prev_proportions = copy.deepcopy(self.proportions)

        tau = copy.deepcopy(np.exp(log_tau))
        nk = tau.sum(axis=1) + 10 * np.finfo(tau.dtype).eps

        #####################
        # Update proportions
        proportions_em = nk / n_samples
        proportions = np.zeros((self.n_components,))
        e_value = np.sum(np.multiply(self.prev_proportions, np.log(self.prev_proportions)))

        for k in range(self.n_components):
            proportions[k] = proportions_em[k] + self.beta * self.prev_proportions[k] * (
                np.log(self.prev_proportions[k]) - e_value
            )
        proportions = proportions / np.sum(proportions)
        self.proportions = copy.deepcopy(proportions)
        ##########################

        ######################################
        # Update beta parameter
        eta = np.amin([1.0, 0.5 ** (np.floor(n_features / 2.0 - 1.0))])
        left = (
            np.sum(np.exp(-eta * n_samples * np.abs(self.proportions - self.prev_proportions)))
            / self.n_components
        )

        max_proportions_em = np.amax(proportions_em)
        max_proportions_old = np.amax(self.prev_proportions)

        if e_value == 0.0:
            right = 0.0
        else:
            right = (1.0 - max_proportions_em) / (-max_proportions_old * e_value)

        self.beta = 1.0 * np.amin([left, right])

        #####################################
        # Annihilation of too small clusters
        mask = self.proportions >= 1.0 / n_samples
        new_k = np.sum(mask)
        self.history.save_variables(np.argwhere(mask)[:, 0], "index_bf_mask")
        if self.n_components != new_k:
            if new_k <= 1:
                raise ValueError("Invalid value for n_components\nEstimation finished")

            self.n_components = new_k

            new_proportions = self.proportions[mask]
            new_proportions = new_proportions / np.sum(new_proportions)
            self.proportions = copy.deepcopy(new_proportions)

            new_log_tau = log_tau[mask, :]
            new_log_tau = new_log_tau - logsumexp(new_log_tau, axis=0)[np.newaxis, :]
            log_tau = copy.deepcopy(new_log_tau)
            tau = np.exp(log_tau)

            e1 = e1[mask, :]
            e2 = e2[mask, :]

            # Delete means corresponding to annihilated clusters
            self.means = self.means[mask, :]
            self.prev_means = self.prev_means[mask, :]
            self.n_iter_kconst_ = 1

        else:
            tau = np.exp(log_tau)
            self.n_iter_kconst_ += 1
        self.history.save_variables(self.n_components, "n_components")
        ###############################################

        ##################################################################################
        # Check for a pathological solution corresponding to superimposed clusters
        # Desactivate the proportions regularisation after a certain number of iterations.
        if (iteration >= self.minimal_iteration_number) and (
            (self.history.n_components[iteration - 100] == self.n_components)
        ):
            patho_clusters = check_patho_clusters(self.means, self.covariances)
            if not patho_clusters:
                self.beta = 0.0
            ######################################
            # Merge superimposed clusters
            ######################################
            elif self.n_iter_kconst_ > 200 and patho_clusters:
                log_tau, _ = self._merge_clusters(log_tau, None)
            else:
                self.minimal_iteration_number += 50
        ##########################################

        #############################################
        # Update probabilities of discrete parameters
        if self.type_discrete_features is not None:
            p_discrete = self._m_step_discrete(x_discr, log_tau)
            self.p_discrete = copy.deepcopy(p_discrete)
        #############################################

        ##########################################################
        # Update alpha parameters, with two different equations,
        # depending on distances of mean parameters to data points
        #####################################################
        # Update alpha when locations are closed to data points
        self.alphas = np.empty((self.n_components, n_features))

        mask_locations_points = np.ones(self.n_components, dtype=int)
        for k in range(self.n_components):
            if (np.sum(np.abs(x_cont - self.means[k]), axis=1) < 1e-14).any():
                it_last = -1
                index_before_delete = self.history.index_bf_mask[it_last][
                    k
                ]  # index dans vecteur précédent
                saved_means_cluster = self.history.means[it_last][index_before_delete, :]
                while (np.sum(np.abs(x_cont - saved_means_cluster), axis=1) < 1e-14).any():
                    it_last = it_last - 1
                    index_before_delete = self.history.index_bf_mask[it_last][index_before_delete]
                    saved_means_cluster = self.history.means[it_last][index_before_delete]

                self.means[k] = copy.deepcopy(saved_means_cluster)
                mask_locations_points[k] = 0
                tau_e1 = np.sum(tau[k] * e1[k])
                self.alphas[k] = np.dot(tau[k], (x_cont - self.means[k])) / tau_e1

        mask_locations_points = mask_locations_points == 1
        #################################################
        # Update alphas when means are not closed to data
        self.alphas[mask_locations_points] = estimate_alpha(
            log_tau[mask_locations_points],
            e1[mask_locations_points],
            e2[mask_locations_points],
            x_cont,
        )

        ##############################
        # Intermediate E-step - multicycle ECM
        try:
            _, log_tau, e1, e2 = self._e_step(x)
        except Exception as e:
            print(e)
        ##############################

        #############################
        # Update covariance matrices
        cov_reg = self._estimate_cov_matrices(x_cont, log_tau, e1, e2, cov_comput)
        self.covariances = copy.deepcopy(cov_reg)

    def fit_predict(self, x, cov_comput="original"):
        """Main method to run Dynamical EM algorithms to estimate
        mixture models with SAL continuous variables.

        Parameters
        ----------
        x : see NOTATIONS.md
        cov_comput : str, optional
            method used for covariance computation, "Fraley" or "Baudry" regularisations availables, by default "original"

        Returns
        -------
        labels : array, shape (n_samples)
            Hard assignments of samples x to estimated classes.
        """

        ############################################
        # Split of continuously and discretly distributed features,
        # creation of new_index_features with init=True
        ################################################
        x_cont, x_discr = self.transform_data(x, init=True)
        x_cont = np.copy(x_cont, order="F")

        # Initialization of parameters and first e-step()
        self.random_state = check_random_state(self.random_state)
        self._initialize_mixture_parameters(x_cont, x_discr)
        log_prob_norm, log_tau, e1, e2 = self._e_step(x)
        ll = log_prob_norm
        self.history.save_variables(ll, "log_likelihood")

        self.converged_ = False
        iteration = 1

        #######################################
        # First iteration - estimation of means
        self.prev_means = copy.deepcopy(self.means)
        self.means = estimate_locations(log_tau, e1, e2, x_cont)
        while True:
            try:
                self._m_step(x, log_tau, e1, e2, iteration, cov_comput)
                self.history.save_variables(
                    [
                        self.proportions,
                        self.means,
                        self.covariances,
                        self.alphas,
                        self.beta,
                        self.p_discrete,
                        self.n_components,
                    ],
                    [
                        "proportions",
                        "means",
                        "covariances",
                        "alphas",
                        "beta",
                        "p_discrete",
                        "n_components",
                    ],
                )

            except Exception as e:
                print("Exception in fit-predict loop m-step")
                print(e)
                if str(e).startswith("Invalid value for n_components"):
                    self.means = copy.deepcopy(self.prev_means)
                    self.proportions = copy.deepcopy(self.prev_proportions)
                break

            try:
                log_prob_norm, log_tau, e1, e2 = self._e_step(x)
                ll = log_prob_norm
                self.history.save_variables(
                    [ll, log_tau.argmax(axis=0)], ["log_likelihood", "labels"]
                )
            except Exception as e:
                print("Error in e-step")
                print(e)
                break

            #################################
            # Update means - next iteration
            self.prev_means = copy.deepcopy(self.means)
            self.means = estimate_locations(log_tau, e1, e2, x_cont)

            #########################################
            # Check convergence with Aitken criterion
            if iteration >= 4:
                loglikelihood_future = self.history.log_likelihood[-1]
                loglikelihood_actual = self.history.log_likelihood[-2]
                loglikelihood_prev = self.history.log_likelihood[-3]
                a_k = (loglikelihood_future - loglikelihood_actual) / (
                    loglikelihood_actual - loglikelihood_prev
                )
                loglikelihood_inf = loglikelihood_actual + (
                    loglikelihood_future - loglikelihood_actual
                ) / (1.0 - a_k)
                loglikelihood_prev2 = self.history.log_likelihood[-4]
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
                if np.abs(loglikelihood_inf - loglikelihood_infprev) < self.eps:  ## Aitken bohning
                    patho_clusters = check_patho_clusters(self.means, self.covariances)
                    if not patho_clusters:
                        self.converged_ = True
                        break
                    iteration += 1
                else:
                    iteration += 1
            else:
                iteration += 1

            self.iteration_temp = iteration

        self.n_iter = iteration
        self.ll = ll

        # Always do a final e-step to guarantee
        # that the labels returned by fit_predict(x)
        # are always consistent with fit(x).predict(x)
        try:
            _, log_tau, _, _ = self._e_step(x)
        except Exception as e:
            print("Final e-step not working")
            print(e)
            print("K = ", self.n_components)

        self.history.save_variables(log_tau.argmax(axis=0), "final_labels")

        return log_tau.argmax(axis=0)

    def fit(self, x, cov_comput="original"):
        """Estimate SAL mixture model parameters with the DEM-MD algorithm.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        cov_comput : str, optional
            Method used for covariance computation, Fraley or Baudry regularisations availables, by default "original"

        Returns
        -------
        self
        """
        self.fit_predict(x, cov_comput)
        return self
