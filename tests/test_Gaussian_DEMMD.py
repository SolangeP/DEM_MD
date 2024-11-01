import os
import sys
import pytest
from scipy import stats

import random
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

sys.path.append("../")
from utils.sampling import simulation_mixed_data

from DEM_MD_gaussian import GaussianDEMMD, estimate_gaussian_log_proba
from base_DEM_MD import BaseDEMMD


class TestClassGaussianDEMMD:
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

    def test_gaussian_mixture_attributes(self):
        n_samples, n_features_all = self.data.shape

        type_qual_feats = ["Poisson", "Multinomial", "Bernoulli"]
        index_qual_feats = [np.array([2]), np.array([3]), np.array([4])]
        new_index_qual_feats = [
            np.array([0]),
            np.array([1, 2, 3]),
            np.array([4]),
        ]
        n_feat_quant, n_feat_qual = n_features_all - len(type_qual_feats), len(type_qual_feats)
        n_modalities_multinomial = self.data[:, 3].max() - self.data[:, 3].min() + 1
        GDEM = GaussianDEMMD(
            is_dummy=False,
            type_discrete_features=type_qual_feats,
            index_discrete_features=index_qual_feats,
        )

        x_cont, x_discr = GDEM.transform_data(self.data, init=True)
        GDEM._initialize_mixture_parameters(x_cont, x_discr)

        assert GDEM.n_components == n_samples
        assert x_discr.shape[1] == n_feat_qual + n_modalities_multinomial - 1
        assert x_cont.shape[1] == n_feat_quant

        assert all(
            [
                (element == new_index_qual_feats[i]).all()
                for i, element in enumerate(GDEM.new_index_discrete_features)
            ]
        )
        assert GDEM.eps == 1e-6  # default value for Gaussian_DEM_MD
        assert GDEM.gamma == 1e-4  # default value for Gaussian_DEM_MD

    def test_gaussian_DEMMD_log_probabilities(self):

        data_cont = self.data[:, :2]
        # precisions = np.zeros((self.n_components, self.n_dim, self.n_dim))
        log_prob_naive = np.empty((self.n_components, self.n_samples))
        for i, (mean, std, prop) in enumerate(zip(self.locations, self.scales, self.proportions)):

            log_prob_naive[i, :] = stats.multivariate_normal.logpdf(data_cont, mean, std) + np.log(
                prop
            )
            # precisions[i] = np.linalg.inv(std)

        log_prob = estimate_gaussian_log_proba(
            data_cont, self.locations, self.scales, self.proportions
        )

        assert_array_almost_equal(log_prob, log_prob_naive)

    def test_gaussian_DEMMD_estimate_log_prob_resp(self):
        data_cont = self.data[:, :2]

        type_qual_feats = ["Poisson", "Multinomial", "Bernoulli"]
        index_qual_feats = [np.array([2]), np.array([3]), np.array([4])]

        GDEM = GaussianDEMMD(
            is_dummy=False,
            type_discrete_features=type_qual_feats,
            index_discrete_features=index_qual_feats,
        )

        GDEM.fit(self.data)
        _, resp = GDEM._e_step(self.data)  # n_components, n_samples
        assert_array_almost_equal(np.exp(resp).sum(axis=0), np.ones(self.n_samples))

    def test_gaussian_mixture_fit_predict(self):
        import copy

        type_qual_feats = ["Poisson", "Multinomial", "Bernoulli"]
        index_qual_feats = [np.array([2]), np.array([3]), np.array([4])]

        GDEM = GaussianDEMMD(
            is_dummy=False,
            type_discrete_features=type_qual_feats,
            index_discrete_features=index_qual_feats,
        )

        # check if fit_predict(X) is equivalent to fit(X).predict(X)
        f = copy.deepcopy(GDEM)
        Y_pred1 = f.fit(self.data).predict(self.data)
        Y_pred2 = GDEM.fit_predict(self.data)
        assert GDEM.n_components == self.n_components
        assert_array_equal(Y_pred1, Y_pred2)
