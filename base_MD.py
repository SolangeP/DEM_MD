import warnings
import inspect
import pickle
import os
import datetime
from itertools import filterfalse
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import logsumexp, factorial
from sklearn.cluster import KMeans
import pandas as pd
from history import Historic
from utils.validation import check_random_state
from utils.calculation import scheme_temperature


def _get_attributes_names(cls):
    """Get parameter names for the estimator
    Parameters
    ----------
    cls : class

    Returns
    -------
    list
        sorted names of class attributes
    """

    attributes = inspect.getmembers(cls, lambda a: not inspect.isroutine(a))
    filtered_attrib = [a for a in attributes if not (a[0].startswith("_") and a[0].endswith("__"))]

    return sorted([p[0] for p in filtered_attrib])


class BaseMixtureMD(metaclass=ABCMeta):
    """Base class for mixture models on mixed-type data.
    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models on mixed type data.
    """

    def __init__(
        self,
        n_components,
        eps,
        reg_cov,
        maxiter,
        init_params,
        warm_start,
        random_state,
        type_discrete_features,
        index_discrete_features,
        is_dummy,
    ):
        self.n_components = n_components
        self.eps = eps
        self.maxiter = maxiter
        self.init_params = init_params
        self.reg_cov = reg_cov
        self.warm_start = warm_start
        self.random_state = random_state
        self.type_discrete_features = type_discrete_features
        self.index_discrete_features = index_discrete_features
        if type_discrete_features is None and index_discrete_features is None:
            pass
        elif type_discrete_features is None or index_discrete_features is None:
            raise ValueError("Need to define type_discrete_features and index_discrete_features.")
        else:
            if len(type_discrete_features) != len(index_discrete_features):
                raise ValueError(
                    "type_discrete_features and index_discrete_features have different sizes."
                )
            if not all(
                e
                in [
                    "Poisson",
                    "Bernoulli",
                    "Multinomial",
                ]
                for e in type_discrete_features
            ):
                raise ValueError("Discrete distribution not implemented.")
            flatten_indexes_qual = [
                indice for arrays in index_discrete_features for indice in arrays
            ]
            if not all(
                [
                    flatten_indexes_qual[i] == flatten_indexes_qual[i - 1] + 1
                    for i in range(1, len(flatten_indexes_qual))
                ]
            ):
                raise AssertionError("Qualitative indexes should be all consecutives.")

        self.is_dummy = is_dummy
        self.history = Historic()  # Object to save all parameters and estimates
        self.ll = -np.infty
        self.n_iter = 0

    def _initialize_pdiscrete(self, x_discr, tau):
        """Initialize the parameters of discrete distributions.

        Parameters
        ----------
        x_discr : see Notations.md
        tau : see Notations.md

        Returns
        -------
        p_discrete : list
            Initial parameter values for discrete distributions
        """
        p_discrete = []

        nk = tau.sum(axis=1) + 10 * np.finfo(tau.dtype).eps

        for type_var, index in zip(self.type_discrete_features, self.new_index_discrete_features):
            if type_var == "Bernoulli":
                p_discrete.append(np.dot(tau, x_discr[:, index]) / nk[:, np.newaxis])
            elif type_var == "Poisson":
                p_discrete.append(np.dot(tau, x_discr[:, index]) / nk[:, np.newaxis])
            elif type_var == "Multinomial":
                p_discrete.append(np.dot(tau, x_discr[:, index]) / nk[:, np.newaxis])

        return p_discrete

    def _initialize_parameters_em(self, x, random_state):
        """Initialization of the model parameters.
        **Used for EM algorithms which require a particular initialization.**
        Dynamical EM algorithms do not rely on this method.

        Parameters
        ----------
        x : see Notations.md
        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = x.shape
        x_cont, x_discr = self.transform_data(x)

        if self.init_params == "kmeans":
            tau = np.zeros((n_samples, self.n_components))
            label = (
                KMeans(n_clusters=self.n_components, n_init=1, random_state=random_state)
                .fit(x_cont)
                .labels_
            )
            tau[np.arange(n_samples), label] = 1

        elif self.init_params == "random":
            tau = random_state.rand(n_samples, self.n_components)
            tau /= tau.sum(axis=1)[:, np.newaxis]

        else:
            raise ValueError(f"Unimplemented initialization method {self.init_params}")
        self._initialize_mixture_parameters(x_cont, x_discr, tau.T)

    def get_params(self):
        """Returns all attributes of the class, as a dictionnary.

        Returns
        -------
        out : dict
        """
        out = {}
        for var in _get_attributes_names(self):
            value = getattr(self, var)
            out[var] = value
        return out

    def transform_data(self, x, init=False):
        """The data set x is transformed to separate continuous and discrete attributes (defined in columns of x).
        Multinomial columns are transformed into dummy matrices and concatenated with other discrete columns.
        If init=True, a new index vector is created to take into account this change of dimensions into the discrete data matrix.

        Parameters
        ----------
        x : see Notations.md
        init : bool, optional
            Used to create the new index vector at first call, by default False

        Returns
        -------
        x_cont : see Notations.md
        x_discr : see Notations.md
            returned if discrete data are present, else an empty list is returned
        """
        _, n_features_all = x.shape

        x_discr = []

        if self.type_discrete_features:
            flatten_indexes_discrete = [
                indice for arrays in self.index_discrete_features for indice in arrays
            ]

            assert all(
                indice < n_features_all for indice in flatten_indexes_discrete
            ), "At least one qualitative feature index is out of data bounds."

            continuous_indexes = [
                *filterfalse(lambda i: i in flatten_indexes_discrete, [*range(n_features_all)])
            ]
            if init:
                n_cont = len(continuous_indexes)
                self.new_index_discrete_features = [
                    self.index_discrete_features[i] - n_cont
                    for i in range(len(self.index_discrete_features))
                ]

            for j, (type_var, index) in enumerate(
                zip(self.type_discrete_features, self.index_discrete_features)
            ):
                if type_var != "Multinomial":
                    x_discr.extend(x[:, index].T)
                elif (type_var == "Multinomial") and not self.is_dummy:
                    dummy_multi = np.array(pd.get_dummies(x[:, index].reshape(-1)))
                    number_mod = dummy_multi.shape[1]
                    if init:
                        self.new_index_discrete_features[j] = np.array(
                            np.arange(
                                self.new_index_discrete_features[j][0],
                                self.new_index_discrete_features[j][0] + number_mod,
                            )
                        )
                        self.new_index_discrete_features[j + 1 :] = [
                            self.new_index_discrete_features[i] + number_mod - 1
                            for i in range(j + 1, len(self.index_discrete_features))
                        ]
                    x_discr.extend(dummy_multi.T)
                elif (type_var == "Multinomial") and self.is_dummy:
                    dummy_multi = x[:, index]
                    number_mod = dummy_multi.shape[1]
                    x_discr.extend(dummy_multi.T)

            x_discr = np.array(x_discr).T
        else:
            continuous_indexes = [i for i in range(n_features_all)]

        x_cont = x[:, continuous_indexes]
        return x_cont, x_discr

    @abstractmethod
    def _initialize_mixture_parameters(self, x_cont, x_discr, tau):
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        x_cont : see Notations.md
        x_discr: see Notations.md
        tau: see Notations.md
        """

    def fit(self, x):
        """Estimate model parameters with the EM algorithm.

        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        self
        """
        self.fit_predict(x)
        return self

    @abstractmethod
    def _estimate_weighted_log_prob(self, x):
        """Estimate the weighted log-probabilities, log P(x | Z) + log weights.

        Parameters
        ----------
            x : see Notations.md
        """

    @abstractmethod
    def _m_step(self, x, log_tau, iteration):
        """Abstract method for M-step.

        Parameters
        ----------
        x : see Notations.md
        log_tau : see Notations.md
        """

    def _m_step_discrete(self, x_discr, log_tau):
        """
        Returns a list with all discrete parameters :
        - an element of the list is an array with parameters of one discrete variable for the n_components clusters.
        - for Bernoulli and Poisson variables : array (K,1)
        - for Multinomial variables : array (K,number_mod)

        Parameters
        ----------
        x_discr : see Notations.md
        log_tau : see Notations.md

        Returns
        -------
        p_discrete : list
        """
        p_discrete = []

        tau = np.exp(log_tau)
        nk = tau.sum(axis=1) + 10 * np.finfo(tau.dtype).eps
        for type_var, index in zip(self.type_discrete_features, self.new_index_discrete_features):
            if type_var == "Bernoulli":
                p_discrete.append(np.dot(tau, x_discr[:, index]) / nk[:, np.newaxis])
            elif type_var == "Poisson":
                p_discrete.append(np.dot(tau, x_discr[:, index]) / nk[:, np.newaxis])
            elif type_var == "Multinomial":
                p_discrete.append(np.dot(tau, x_discr[:, index]) / nk[:, np.newaxis])
        return p_discrete

    def _estimate_discrete_log_prob(self, x_discr):
        """
        Estimate the weighted discrete log-probabilities, log P(x_discr | Z).

        Parameters
        ----------
        x_discr : see Notations.md

        Returns
        -------
        log_tau : array-like, shape (n_components, n_samples)

        """

        n_samples, _ = x_discr.shape

        log_tau = np.zeros((self.n_components, n_samples))
        for j, (type_var, index) in enumerate(
            zip(self.type_discrete_features, self.new_index_discrete_features)
        ):
            for k in range(self.n_components):
                try:
                    if type_var == "Bernoulli":
                        index_val = index[0]
                        if (
                            np.isclose(self.p_discrete[j][k][0], 1.0, atol=1e-16, rtol=1e-15)
                        ) or np.isclose(self.p_discrete[j][k][0], 0.0, atol=0.0):
                            pass
                        else:
                            log_tau[k, :] += x_discr[:, index_val].reshape(n_samples) * np.log(
                                self.p_discrete[j][k]
                            ) + (1 - x_discr[:, index_val].reshape(n_samples)) * np.log(
                                1 - self.p_discrete[j][k]
                            )
                    elif type_var == "Poisson":
                        index_val = index[0]
                        if (
                            np.isclose(self.p_discrete[j][k], 1.0, atol=1e-16, rtol=1e-15)
                        ).any() or (np.isclose(self.p_discrete[j][k], 0.0, atol=0.0).any()):
                            pass
                        else:
                            log_tau[k, :] += (
                                -self.p_discrete[j][k]
                                + np.log(self.p_discrete[j][k]) * (x_discr[:, index_val])
                                - np.log(factorial(x_discr[:, index_val]))
                            )
                    elif type_var == "Multinomial":
                        if (
                            np.isclose(self.p_discrete[j][k], 1.0, atol=1e-16, rtol=1e-15)
                        ).any() or (np.isclose(self.p_discrete[j][k], 0.0, atol=0.0).any()):
                            pass
                        else:
                            log_tau[k, :] += np.dot(
                                x_discr[:, index], np.log(self.p_discrete[j][k])
                            )

                except np.linalg.LinAlgError:
                    print("In estimate_discrete_log_prob:")
                    print("Number of components", self.n_components)
                    print("Covariances", self.covariances[k])
                    break
        return log_tau

    def _e_step(self, x):
        """E step

        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        mean(log_prob) : Average of logarithms of probabilities for each sample in data

        log_resp : array, shape (n_components, n_samples)
            Log of posterior probabilities (or responsabilities) of each sample in data.

        """

        log_prob_norm, log_resp = self._estimate_log_prob_resp(x)

        return np.mean(log_prob_norm), log_resp

    def _estimate_log_prob_resp(self, x):
        """Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per component and responsibilities for each sample in x with respect to the current state of the model.

        Parameters
        ----------
        x : see Notations.md
        Returns
        -------
        log_prob : Sum of logarithms of probabilities of each sample in x
            Sum( log p(x))

        log_resp : array, shape (n_components, n_samples)
            Log of posterior probabilities (or responsabilities) of each sample in x.

        """

        weighted_log_prob = self._estimate_weighted_log_prob(x)

        log_prob_norm = logsumexp(weighted_log_prob, axis=0)

        return log_prob_norm, weighted_log_prob - log_prob_norm[np.newaxis, :]

    def fit_predict(self, x):
        """Main function which runs any EM algorithm for mixture estimation.
        Overrided for DEM algorithms by child classes.

        Parameters
        ----------
        x : see Notations.md
        Returns
        -------
        labels : array of shape (n_samples)
            Array with had assignement of each sample to a cluster k
        """

        x_cont, _ = self.transform_data(x, init=True)
        _, n_features = x_cont.shape
        self.n_features = n_features

        random_state = check_random_state(self.random_state)
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        self.converged_ = False

        if do_init:
            self._initialize_parameters_em(x, random_state)
        else:
            print("Warm start so no initialisation")

        ll = -np.infty if do_init else self.ll

        for iteration in range(1, self.maxiter + 1):
            log_prob_norm, log_tau = self._e_step(x)
            self._m_step(x, log_tau, iteration)

            ll = log_prob_norm

            self.history.save_variables([ll, log_tau.argmax(axis=0)], ["log_likelihood", "labels"])

            ######################################
            # Stopping criterion: Aitken's acceleration
            ######################################
            if iteration >= 4:

                small_list_ll = [
                    self.history.log_likelihood[-i] for i in reversed(range(1, 5))
                ]  # ordered as [ll[-4],ll[-3],ll[-2],ll[-1]]

                if (small_list_ll[2] - small_list_ll[1]) == 0.0:
                    a_k = 0.0
                else:
                    a_k = (small_list_ll[3] - small_list_ll[2]) / (
                        small_list_ll[2] - small_list_ll[1]
                    )
                loglikelihood_inf = small_list_ll[2] + (small_list_ll[3] - small_list_ll[2]) / (
                    1.0 - a_k
                )
                if (small_list_ll[1] - small_list_ll[0]) == 0.0:
                    a_kprev = 0.0
                else:
                    a_kprev = (small_list_ll[2] - small_list_ll[1]) / (
                        small_list_ll[1] - small_list_ll[0]
                    )
                loglikelihood_infprev = small_list_ll[1] + (small_list_ll[2] - small_list_ll[1]) / (
                    1.0 - a_kprev
                )
            else:
                loglikelihood_inf = 10.0
                loglikelihood_infprev = 50.0

            if np.abs(loglikelihood_inf - loglikelihood_infprev) < self.eps:
                self.converged_ = True
                break

        if not self.converged_:
            warnings.warn(
                "Algorithm  did not converge. "
                "Try different init parameters, "
                "or increase maxiter, tol "
                "or check for degenerate data."
            )

        self.n_iter = iteration
        self.ll = ll

        # Always do a final e-step to guarantee that
        # the labels returned by fit_predict(x)
        #  are always consistent with fit(x).predict(x)
        _, log_tau = self._e_step(x)
        self.history.save_variables(log_tau.argmax(axis=0), "final_labels")
        return log_tau.argmax(axis=0)

    def score_samples(self, x):
        """Compute the weighted log probabilities for each sample.
        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in x.
        """
        return logsumexp(self._estimate_weighted_log_prob(x), axis=0)

    def score(self, x):
        """Compute the per-sample average log-likelihood of the given data x.
        Parameters
        ----------
            x : see Notations.md

        Returns
        -------
        log_likelihood : float
            Log likelihood of the Gaussian mixture given x.
        """
        return self.score_samples(x).mean()

    def predict(self, x):
        """Predict the labels for the data samples in x using trained model.
        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        if not hasattr(self, "n_iter"):
            raise ValueError("Model was not fitted")

        if hasattr(self, "use_temp") and self.use_temp:
            self.temperature = scheme_temperature(
                self.iteration_temp, b=self.temp_b, rb=self.temp_rb
            )
            weighted_log_prob = self._estimate_weighted_log_prob(x) * (1.0 / self.temperature)
        else:
            weighted_log_prob = self._estimate_weighted_log_prob(x)
        normalized_weighted = (
            weighted_log_prob - logsumexp(weighted_log_prob, axis=0)[np.newaxis, :]
        )
        return normalized_weighted.argmax(axis=0)

    def export_history(self, save_file=False, path="./", name=None):
        """Method exporting several attributes and estimates into a dictionnary.

        Parameters
        ----------
        save : bool, optional
            if True, directly exports the dictionary into a .py file at indicated path, by default False
        path : str, optional
            path to the desired location for saving the dictionnary, by default "./"
        name : str, optional
            name of the file, is "result" if None, by default None

        Returns
        -------
        dico : dict
        """

        for arg in [
            "converged_",
            "cov_init",
            "n_iter",
            "limited",
            "warm_start",
            "init_params",
            "maxiter",
            "n_components",
            "means_init",
            "pi_init",
        ]:
            if hasattr(self, arg) and not hasattr(self.history, arg):
                if getattr(self, arg) is not None:
                    self.history.save_variables(getattr(self, arg), arg)

        dico = self.history.__dict__

        if save_file:
            is_exist = os.path.exists(path)
            if not is_exist:
                os.makedirs(path)

            date = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            if name:
                f = open(path + name + f"_{date}.py", "wb")
            else:
                f = open(path + f"/result_{date}.py", "wb")
            pickle.dump(dico, f)
            f.close()

        return dico

    def bic(self, x):
        """Bayesian information criterion for the current model on the input x.

        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        bic : float
        """
        return -self.score(x) * x.shape[0] + self._n_parameters() / 2.0 * np.log(x.shape[0])

    def icl(self, x):
        """Integrated Complete Likelihood criterion for the current model on the input x.

        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        icl: float
        """
        log_tau = self._estimate_log_prob_resp(x)[1]
        mean_entropy = -np.sum(np.multiply(np.exp(log_tau), log_tau))
        return self.bic(x) + mean_entropy

    def nec(self, x, l1):
        """Negative Entropy criterion for the current model on the input x.

        Parameters
        ----------
        x : see Notations.md
        l1 : float
            required likelihood of a model with 1 component on the current input x.

        Returns
        -------
        nec : float
        """
        log_tau = self._estimate_log_prob_resp(x)[1]
        e_k = -np.sum(np.multiply(np.exp(log_tau), log_tau))
        c_k = np.sum(np.multiply(self._estimate_weighted_log_prob(x), np.exp(log_tau)))
        l_k = c_k + e_k
        return e_k / (l_k - l1)
