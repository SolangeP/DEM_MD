import os
import sys
import pytest
from scipy import stats

import random
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

sys.path.append("../")
from utils.sampling import simulation_mixed_data
from DEM_MD_sal import SALDEMMD, estimate_sal_log_proba
from base_DEM_MD import BaseDEMMD


class TestClassSALDEMMD:
    random.seed(1)

    n_components = 2
    n_dim = 2
    n_samples = 400
    random_state = 1
    ### Parameters for SAL distributions
    proportions = np.array([1.0 / 4.0, 3.0 / 4.0])
    locations = np.array([[-6, 1], [4, 2]])
    scales = np.array([[[1, 0], [0, 3]], [[2, 0.5], [0.5, 4]]])
    alphas = np.array([[-3, 2], [1, 4]])

    ### Parameters for discrete distributions
    Bernoulli0 = np.array([0.65, 0.35])
    Multinomial1 = np.array([[0.2, 0.4, 0.2], [0.3, 0.1, 0.6]])
    Poisson2 = np.array([100, 22])

    args = ["proportions", "locations", "scales", "alphas"]
    dict_cont_sal = {i: eval(i) for i in args}

    args = ["Bernoulli0", "Multinomial1", "Poisson2"]
    dict_discr_sal = {i: eval(i) for i in args}

    type_qual_feats = ["Bernoulli", "Multinomial", "Poisson"]
    index_qual_feats = [np.array([2]), np.array([3]), np.array([4])]

    data, true_labels = simulation_mixed_data(
        "SAL", type_qual_feats, dict_cont_sal, dict_discr_sal, n_samples
    )

    def test_sal_mixture_attributes(self):
        from utils.validation import check_random_state

        n_samples, n_features_all = self.data.shape

        new_index_qual_feats = [
            np.array([0]),
            np.array([1, 2, 3]),
            np.array([4]),
        ]
        n_feat_quant, n_feat_qual = n_features_all - len(self.type_qual_feats), len(
            self.type_qual_feats
        )
        n_modalities_multinomial = self.data[:, 3].max() - self.data[:, 3].min() + 1
        SDEM = SALDEMMD(
            type_discrete_features=self.type_qual_feats,
            index_discrete_features=self.index_qual_feats,
            random_state=self.random_state,
        )

        x_cont, x_discr = SDEM.transform_data(self.data, init=True)
        SDEM.random_state = check_random_state(self.random_state)
        SDEM._initialize_mixture_parameters(x_cont, x_discr)

        assert SDEM.n_components == n_samples
        assert x_discr.shape[1] == n_feat_qual + n_modalities_multinomial - 1
        assert x_cont.shape[1] == n_feat_quant

        assert all(
            [
                (element == new_index_qual_feats[i]).all()
                for i, element in enumerate(SDEM.new_index_discrete_features)
            ]
        )
        assert SDEM.eps == 1e-4  # default value for SAL_DEM_MD
        assert SDEM.gamma == 0.0  # default value for SAL_DEM_MD
        assert SDEM.use_temp == True  # default value for SAL_DEM_MD
        assert SDEM.temp_b == 1.0  # default value for SAL_DEM_MD
        assert SDEM.temp_rb == 3.0  # default value for SAL_DEM_MD

    def test_sal_DEMMD_estimate_log_prob_resp(self):

        SALDEM = SALDEMMD(
            type_discrete_features=self.type_qual_feats,
            index_discrete_features=self.index_qual_feats,
        )

        SALDEM.fit(self.data)
        _, resp, _, _ = SALDEM._e_step(self.data)  # n_components, n_samples
        assert_array_almost_equal(np.exp(resp).sum(axis=0), np.ones(self.n_samples))

    def test_sal_mixture_fit_predict(self):
        import copy

        index_qual_feats = [np.array([2]), np.array([3]), np.array([4])]

        SALDEM = SALDEMMD(
            random_state=self.random_state,
            type_discrete_features=self.type_qual_feats,
            index_discrete_features=index_qual_feats,
        )

        # check if fit_predict(X) is equivalent to fit(X).predict(X)
        f = copy.deepcopy(SALDEM)
        Y_pred1 = f.fit(self.data).predict(self.data)
        Y_pred2 = SALDEM.fit_predict(self.data)
        assert SALDEM.n_components == self.n_components
        assert f.n_components == self.n_components
        assert_array_equal(Y_pred1, Y_pred2)
