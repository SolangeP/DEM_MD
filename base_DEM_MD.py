from abc import abstractmethod
import copy
import itertools
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.special import logsumexp
from sklearn.metrics import pairwise_distances_chunked
from base_MD import BaseMixtureMD


class BaseDEMMD(BaseMixtureMD):
    """Base class for Dynamical EM to estimate mixture models on mixed data.
    This abstract class specifies an interface for all robust mixture classes and
    provides basic common methods for robust mixture models.
    It inherits from BaseMixedMixture class which is Abstract class for dynamical and non-dynamical EM.
    """

    def __init__(
        self,
        eps,
        random_state,
        is_dummy,
        type_discrete_features,
        index_discrete_features,
    ):
        super().__init__(
            n_components=0,  # not used in DEM algorithms
            reg_cov=0.0,  # not used in DEM algorithms
            maxiter=0,  # not used in DEM algorithms
            init_params="dynamic",  # not used in DEM initialization
            warm_start=False,  # not used in DEM algorithms
            eps=eps,
            random_state=random_state,
            is_dummy=is_dummy,
            type_discrete_features=type_discrete_features,
            index_discrete_features=index_discrete_features,
        )

    def _initialize_pdiscrete(self, x_discr, tau=None):
        """_summary_

        Parameters
        ----------
        x_discr : see Notations.md

        Returns
        -------
        p_discrete : list
            Initial parameter values for discrete distributions
        """
        p_discrete = []
        for index in self.new_index_discrete_features:
            p_discrete.append(x_discr[:, index])
        return p_discrete

    def _initialize_covariances(self, x):
        """Initialize covariance matrices

        Parameters
        ----------
        x : see Notations.md

        Returns
        -------
        covariances : see Notations.md
        """

        n_samples, n_features = x.shape
        covariances = np.empty((self.n_components, n_features, n_features), dtype=np.float64)
        index_sqrt = int(np.ceil(np.sqrt(self.n_components)) - 1)

        if n_samples <= 10000:
            mat = pdist(x, metric="sqeuclidean")
            mat_nozero = mat[mat > 0.0]
            self.min_dist = np.amin(mat_nozero)

            for k in range(self.n_components):
                i_end = n_samples
                j = k
                i_start = 1

                index_condensed_end = n_samples * j - j * (j + 1) // 2 + i_end - 1 - j
                index_condensed_start = n_samples * j - j * (j + 1) // 2 + i_start - 1 - j
                d_mat = mat[index_condensed_start:index_condensed_end]
                sorted_dist = np.sort(d_mat[d_mat > 0.0])
                covariances[k] = sorted_dist[index_sqrt] * np.eye(n_features, dtype=np.float64)
        else:
            ##########################
            # Speed up by computing on chunks of samples
            gen = pairwise_distances_chunked(x, n_jobs=-1)
            len_total = 0
            self.min_dist = np.infty
            for mat in gen:
                actual_amin = np.amin(mat[mat > 0.0])
                if actual_amin < self.min_dist:
                    self.min_dist = actual_amin
                for k in np.arange(len_total, len_total + (mat).shape[0]):
                    sorted_dist = np.sort(mat[k - len_total][mat[k - len_total] > 0.0])
                    covariances[k] = sorted_dist[index_sqrt] * np.eye(n_features, dtype=np.float64)
                len_total += mat.shape[0]
        return covariances

    @abstractmethod
    def _estimate_weighted_log_prob(self, x):
        """Estimate the weighted log-probabilities, log P(x | Z) + log weights.

        Parameters
        ----------
            x : see Notations.md
        """
        pass

    def _merge_clusters(self, log_tau, gamma_u):
        """_summary_

        Parameters
        ----------
        log_tau : see Notations.md
        gamma_u : see Notations.md

        Returns
        -------
        new_log_tau
            log probabilities after merge of clusters
        new_gamma_u
            probabilities of latent u after merge of clusters, for RobustStudentMixtureDiscrete class
        """

        tau = np.exp(log_tau)
        dist_centroids = cdist(self.means, self.means, metric="euclidean")
        dist_cov = np.zeros((self.n_components, self.n_components))

        for i, j in itertools.combinations(range(self.n_components), 2):
            dist_cov[i, j] = dist_cov[j, i] = np.linalg.norm(
                (self.covariances[i] - self.covariances[j])
            )

        dist_totale = dist_cov + dist_centroids
        index_null = np.argwhere(dist_totale == 0)

        mask_superimposed = index_null[:, 0] != index_null[:, 1]
        index_superimposed = index_null[mask_superimposed]
        index_superimposed = np.unique(np.sort(index_superimposed, axis=1), axis=0)

        ###################################
        # Merge by summing proportions and
        # responsabilities over superimposed clusters
        proportions_merged = copy.deepcopy(self.proportions)
        tau_merged = copy.deepcopy(tau)
        visited_index = []
        for couple_clusters in range(len(index_superimposed)):
            if index_superimposed[couple_clusters, 0] in visited_index:
                pass
            else:
                proportions_merged[index_superimposed[couple_clusters, 0]] = (
                    proportions_merged[index_superimposed[couple_clusters, 0]]
                    + proportions_merged[index_superimposed[couple_clusters, 1]]
                )
                tau_merged[index_superimposed[couple_clusters, 0], :] = (
                    tau_merged[index_superimposed[couple_clusters, 0], :]
                    + tau_merged[index_superimposed[couple_clusters, 1], :]
                )

                visited_index.append(index_superimposed[couple_clusters, 1])

                self.n_components -= 1

        ######### Delete superimposed components ###############
        proportions_merged = np.delete(proportions_merged, visited_index, axis=0)
        proportions_merged = proportions_merged / np.sum(
            proportions_merged, axis=0, dtype=np.float64
        )
        self.proportions = copy.deepcopy(proportions_merged)

        ######### Delete superimposed locations (self.means) and previous locations ###############
        self.means = np.delete(self.means, visited_index, axis=0)
        self.prev_means = np.delete(self.prev_means, visited_index, axis=0)

        if hasattr(self, "dofs"):
            self.dofs = np.delete(self.dofs, visited_index, axis=0)
            new_gamma_u = np.delete(np.copy(gamma_u), visited_index, axis=0)

        tau_merged = np.delete(tau_merged, visited_index, axis=0)
        new_log_tau = np.log(tau_merged)
        new_log_tau = new_log_tau - logsumexp(new_log_tau, axis=0)[np.newaxis, :]

        if hasattr(self, "dofs"):
            return new_log_tau, new_gamma_u
        return new_log_tau
