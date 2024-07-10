"""Base class for mixture models  """

from abc import ABCMeta, abstractmethod
from sklearn.cluster import KMeans
import sklearn.mixture as sm
from scipy.special import logsumexp
import warnings
import inspect
import pickle
import datetime
from scipy.spatial.distance import cdist, pdist
import numpy as np
from history import Historic
from utils.validation import check_random_state


def _get_attributes_names(cls):
    """    Get parameter names for the estimator

    Returns
    -------
    list        
        sorted names of class attributes
    """

    attributes = inspect.getmembers(cls, lambda a: not (inspect.isroutine(a)))
    filtered_attrib = [
        a for a in attributes if not (a[0].startswith("_") and a[0].endswith("__"))
    ]

    # introspect the constructor arguments to find the model parameters
    # to represent
    # init_signature = inspect.signature(init)
    # Consider the constructor parameters excluding 'self'
    # parameters = [p for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    # Extract and sort argument names excluding 'self'
    return sorted([p[0] for p in filtered_attrib])


class BaseMixture(metaclass=ABCMeta):
    """Base class for mixture models.
    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """

    def __init__(
        self, n_components, eps, reg_cov, maxiter, init_params, warm_start, random_state
    ):
        self.n_components = n_components
        self.eps = eps
        self.maxiter = maxiter
        self.init_params = init_params
        self.reg_cov = reg_cov
        self.warm_start = warm_start
        self.random_state = random_state
        self.history = Historic()  ### Object to save all parameters

    def _initialize_parameters(self, x, random_state):
        """Initialize the model parameters.
        Parameters
        ----------
        x : array-like, shape  (n_samples, n_features)
        random_state : RandomState
                A random number generator instance that controls the random seed
                used for the method chosen to initialize the parameters.
        """
        n_samples, _ = x.shape

        if self.init_params == "kmeans":
            tau = np.zeros((n_samples, self.n_components))
            label = (
                KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(x)
                .labels_
            )
            tau[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            tau = random_state.rand(n_samples, self.n_components)
            tau /= tau.sum(axis=1)[:, np.newaxis]

        elif self.init_params == "robust":
            tau = np.eye(n_samples)
        elif self.init_params == "fj":
            highest_ll = -np.inf
            for _ in range(10):
                gm = sm.GaussianMixture(
                    n_components=self.n_components, max_iter=5, init_params="random"
                ).fit(x)
                gm_score = gm.score(x)
                if gm_score > highest_ll:
                    tau = np.exp(gm._estimate_log_prob_resp(x)[1])
                    highest_ll = gm_score
        else:
            raise ValueError(
                "Unimplemented initialization method {}".format(self.init_params)
            )

        self._initialize(x, tau.T)

    def get_params(self):
        """_summary_

        Returns
        -------
        out : dictionary
            _description_
        """
        out = {}
        for var in _get_attributes_names(self):
            value = getattr(self, var)
            # if  hasattr(value, 'get_params'):
            # 	deep_items = value.get_params().items()
            # 	out.update((var + '__' + k, val) for k, val in deep_items)
            out[var] = value

        return out

    @abstractmethod
    def _initialize(self, x, tau):
        """Initialize the model parameters of the derived class.
        Parameters
        ----------
        x : array-like, shape  (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
        """
        pass

    def fit(self, x):
        """Estimate model parameters with the EM algorithm.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
                List of n_features-dimensional data points. Each row
                corresponds to a single data point.
        Returns
        -------
        self
        """
        self.fit_predict(x)
        return self

    @abstractmethod
    def _m_step(self, x, log_tau):
        """M step.
        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
                Logarithm of the posterior probabilities (or responsibilities) of
                the point of each sample in x.
        """
        pass

    @abstractmethod
    def _estimate_weighted_log_prob(self, x):
        """Estimate the weighted log-probabilities, log P(x | Z) + log weights.

        Parameters
        ----------
                x : array-like, shape (n_samples, n_features)
        Returns
        -------
                weighted_log_prob : array, shape (n_component,n_samples)

        """

        pass

    def _e_step(self, x):
        """E step

        Parameters
        ----------
        x : array-like, shape (n_samples,n_features)

        Returns
        -------
        mean(log_prob) : Average of logarithms of probabilities of each sample in data

        log_resp : array, shape (n_components, n_samples)
                Log of posterior probabilities (or responsabilities) of each sample in data.

        """

        log_prob_norm, log_resp = self._estimate_log_prob_resp(x)

        return np.mean(log_prob_norm), log_resp

    def _estimate_log_prob_resp(self, x):
        """Estimate log probabilities and responsibilities for each sample.

                Compute the log probabilities, weighted log probabilities per
                component and responsibilities for each sample in x with respect to
                the current state of the model.

        Parameters
        ----------

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
        n_samples, n_features = x.shape
        self.n_features = n_features
        random_state = check_random_state(self.random_state)
        do_init = not (self.warm_start and hasattr(self, "converged_"))

        self.converged_ = False

        if do_init:
            self._initialize_parameters(x, random_state)
        else:
            print("Warm start so no initialisation")

        LL = -np.infty if do_init else self.LL

        for iteration in range(1, self.maxiter + 1):
            prev_LL = LL

            log_prob_norm, log_tau = self._e_step(x)
            self._m_step(x, log_tau)
            LL = log_prob_norm
            if np.isnan(LL):
                print(iteration)
            change = LL - prev_LL

            self.history.save_variables(LL, "log_likelihood")
            self.history.save_variables(log_tau.argmax(axis=0), "labels")

            if np.abs(change) < self.eps:
                self.converged_ = True
                break

        if not self.converged_ and not self.limited:
            warnings.warn(
                "Algorithm  did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data."
            )

        self.n_iter = iteration
        self.LL = LL

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(x) are always consistent with fit(x).predict(x)
        _, log_tau = self._e_step(x)

        self.history.save_variables(log_tau.argmax(axis=0), "labels_finals")

        # Labels
        return log_tau.argmax(axis=0)

    def score_samples(self, x):
        """Compute the weighted log probabilities for each sample.
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
                List of n_features-dimensional data points. Each row
                corresponds to a single data point.
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
        x : array-like of shape (n_samples, n_dimensions)
                List of n_features-dimensional data points. Each row
                corresponds to a single data point.
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
        x : array-like, shape (n_samples, n_features)
                List of n_features-dimensional data points. Each row
                corresponds to a single data point.
        Returns
        -------
        labels : array, shape (n_samples,)
                Component labels.
        """
        if not (hasattr(self, "n_iter")):
            raise ValueError("Model was not fitted")

        return self._estimate_weighted_log_prob(x).argmax(axis=0)

    def export_history(self, save=False, path="./"):
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
            "proportions_init",
        ]:
            if hasattr(self, arg) and not (hasattr(self.history, arg + "_list")):
                if getattr(self, arg) is not None:
                    self.history.save_variables(getattr(self, arg), arg)

        dico = self.history.__dict__

        if save:
            date = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            f = open(path + "../results/result_{0}.py".format(date), "wb")
            pickle.dump(dico, f)
            f.close()

        return dico

    def _check_patho_clusters(self):
        """ Check if there are pathological clusters,
        which means : same centroids and same covariances

        Returns
        -------
        _type_
            _description_
        """

        patho = False
        dist_centroids = cdist(self.means, self.means, metric="euclidean")
        dist_cov = np.zeros((self.n_components, self.n_components))

        for i in range(self.n_components):
            for j in range(i, self.n_components):
                dist_cov[i, j] = np.linalg.norm(
                    (self.covariances[i] - self.covariances[j])
                )
                dist_cov[j, i] = np.linalg.norm(
                    (self.covariances[j] - self.covariances[i])
                )

        dist_totale = dist_cov + dist_centroids
        index_null = np.argwhere(dist_totale == 0)

        if (len(index_null)) > self.n_components:
            patho = True

        return patho
