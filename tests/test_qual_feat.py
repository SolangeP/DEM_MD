import os
import sys
import pytest

# Fix random seed
import random
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import stats

sys.path.append("../")
from utils.sampling import simulation_mixed_data
from base_DEM_MD import BaseDEMMD
from pandas import get_dummies


class DEM_MD_tests(BaseDEMMD):
    def __init__(
        self,
        eps=0.0,
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

    def _estimate_weighted_log_prob(self, x):
        raise Exception

    def _m_step(self, x, log_tau, iteration):
        raise Exception

    def _initialize_mixture_parameters(self, x_cont, x_discr, tau):
        raise Exception


class TestClassQualFeat:
    random.seed(1)

    n_components = 2  # or K, the number of components
    n_dim = 2  # or c, the number of continuous dimensions
    n_samples = 400  # or n, the size of the dataset

    ### Parameters for Gaussian distributions
    proportions = np.array([1.0 / 4.0, 3.0 / 4.0])
    locations = np.array([[-6, 1], [4, 2]])
    scales = np.array([[[1, 0], [0, 3]], [[2, 0.5], [0.5, 4]]])

    ### Parameters for discrete distributions
    Multinomial0 = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
    Poisson1 = np.array([30, 80])
    Bernoulli2 = np.array([0.2, 0.7])

    args = ["proportions", "locations", "scales"]
    dict_cont_gaussian = {i: eval(i) for i in args}

    args = ["Multinomial0", "Poisson1", "Bernoulli2"]
    dict_discr_gaussian = {i: eval(i) for i in args}

    data, true_labels = simulation_mixed_data(
        "Gaussian",
        ["Multinomial", "Poisson", "Bernoulli"],
        dict_cont_gaussian,
        dict_discr_gaussian,
        n_samples,
    )
    data[:, [3, 2, 4]] = data[:, [2, 3, 4]]  # switch data

    def test_no_index_qual_feat(self):
        type_feat_cat = ["Multinomial"]
        with pytest.raises(ValueError):
            GDEM = DEM_MD_tests(type_discrete_features=type_feat_cat, index_discrete_features=None)

    def test_notmatching_len_attributes(self):
        index_discrfeat_fit = [np.array([3]), np.array([4])]
        type_feat_cat = ["Multinomial"]
        with pytest.raises(ValueError):
            DEM = DEM_MD_tests(
                type_discrete_features=type_feat_cat, index_discrete_features=index_discrfeat_fit
            )

    def test_matching_len_attributes(self):
        index_discrfeat_fit = [np.array([2]), np.array([3])]
        type_feat_cat = ["Poisson", "Multinomial"]
        DEM = DEM_MD_tests(
            type_discrete_features=type_feat_cat, index_discrete_features=index_discrfeat_fit
        )

    def test_no_qual_feat(self):
        DEM = DEM_MD_tests(type_discrete_features=None, index_discrete_features=None)

    def test_qual_feat_oor(self):
        index_discrfeat_fit = [np.array([10])]
        type_feat_cat = ["Multinomial"]

        with pytest.raises(AssertionError):
            DEM = DEM_MD_tests(
                type_discrete_features=type_feat_cat, index_discrete_features=index_discrfeat_fit
            )  # index out of range
            DEM.transform_data(self.data, init=True)

    def test_non_existing_discr_distrib(self):
        index_discrfeat_fit = [np.array([2])]
        type_feat_cat = ["NegativeBinomial"]
        with pytest.raises(ValueError):
            DEM = DEM_MD_tests(
                type_discrete_features=type_feat_cat, index_discrete_features=index_discrfeat_fit
            )

    def test_creating_dummy_var(self):
        type_qual_mult = ["Multinomial"]
        index_qual_mult = [np.array([2])]
        subset_data = self.data[:, [0, 1, 3]]
        n_val_multinomial = int(subset_data[:, 2].max() - subset_data[:, 2].min() + 1)

        DEM = DEM_MD_tests(
            type_discrete_features=type_qual_mult, index_discrete_features=index_qual_mult
        )
        _, x_discr = DEM.transform_data(subset_data, init=True)
        assert (
            DEM.new_index_discrete_features[0].shape[0] == n_val_multinomial
        ), "New index discrete features for 1st qualitative feature should have a size equal to number of modalities of Multinomial feature."
        assert (
            x_discr.shape[1] == n_val_multinomial
        ), "x_discr matrix is dummy matrix of Multinomial feature."

    def test_newindexfeats(self):
        type_qual_feats = ["Poisson", "Multinomial", "Bernoulli"]
        index_qual_feats = [np.array([2]), np.array([3]), np.array([4])]
        new_index_feats = [np.array([0]), np.array([1, 2, 3]), np.array([4])]

        DEM = DEM_MD_tests(
            type_discrete_features=type_qual_feats, index_discrete_features=index_qual_feats
        )
        _, x_discr = DEM.transform_data(self.data, init=True)
        assert isinstance(
            DEM.new_index_discrete_features, list
        ), "New index discrete features should be a list."
        assert all(
            [isinstance(i, np.ndarray) for i in DEM.new_index_discrete_features]
        ), "Each index of new index discrete features should be an ndarray."
        assert all(
            [
                all(DEM.new_index_discrete_features[i] == new_index_feats[i])
                for i in range(len(DEM.new_index_discrete_features))
            ]
        ), "Each index of new index discrete features should be an ndarray."

    def test_DEMMD_log_probabilities_qualfeats(self):

        type_qual_feats = ["Poisson", "Multinomial", "Bernoulli"]
        index_qual_feats = [np.array([2]), np.array([3]), np.array([4])]
        list_params_discr_distribs = [
            np.array([[30], [80]]),
            np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]),
            np.array([[0.2], [0.7]]),
        ]

        log_prob_naive = np.zeros((self.n_components, self.n_samples))
        for j, (type_var, index_qual) in enumerate(zip(type_qual_feats, index_qual_feats)):
            for k in range(self.n_components):
                if type_var == "Multinomial":
                    x_dummy_multi = np.array(
                        get_dummies(self.data[:, index_qual].reshape(self.n_samples))
                    )
                    log_prob_naive[k, :] += stats.multinomial.logpmf(
                        x_dummy_multi,
                        1,
                        list_params_discr_distribs[j][k],
                    )
                elif type_var == "Poisson":
                    log_prob_naive[k, :] += stats.poisson.logpmf(
                        self.data[:, index_qual], list_params_discr_distribs[j][k]
                    ).reshape(self.n_samples)
                elif type_var == "Bernoulli":
                    log_prob_naive[k, :] += stats.bernoulli.logpmf(
                        self.data[:, index_qual], list_params_discr_distribs[j][k]
                    ).reshape(self.n_samples)

        DEM = DEM_MD_tests(
            is_dummy=False,
            type_discrete_features=type_qual_feats,
            index_discrete_features=index_qual_feats,
        )

        x_cont, x_discr = DEM.transform_data(self.data, init=True)
        DEM.n_components = self.n_components
        DEM.p_discrete = list_params_discr_distribs
        log_prob_resp_qual = DEM._estimate_discrete_log_prob(x_discr)

        assert_array_almost_equal(log_prob_resp_qual, log_prob_naive)
