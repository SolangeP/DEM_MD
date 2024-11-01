import os
import sys
import pytest
from scipy import stats

import random
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

sys.path.append("../")
from utils.sampling import simulation_mixed_data

from DEM_MD_student import StudentDEMMD, estimate_student_log_proba
from base_DEM_MD import BaseDEMMD


class TestClassStudentDEMMD:
    random.seed(1)

    n_components = 3  # or K, the number of components
    n_dim = 2  # or c, the number of continuous dimensions
    n_samples = 400  # or n, the size of the dataset

    ### Parameters for Student distributions
    proportions = np.array([1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0])
    locations = np.array([[-6, 1], [4, 5], [3, -3]])
    scales = np.array([[[1, 0], [0, 3]], [[2, 0.5], [0.5, 4]], [[1, 0.5], [0.5, 1]]])
    dofs = np.array([3, 6, 4])

    ### Parameters for discrete distributions
    Poisson0 = np.array([35, 80, 40])
    Multinomial1 = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.4, 0.5]])
    Bernoulli2 = np.array([0.2, 0.4, 0.3])

    args = ["proportions", "locations", "scales", "dofs"]
    dict_cont_student = {i: eval(i) for i in args}

    args = ["Poisson0", "Multinomial1", "Bernoulli2"]
    dict_discr_student = {i: eval(i) for i in args}

    data, true_labels = simulation_mixed_data(
        "Student",
        ["Poisson", "Multinomial", "Bernoulli"],
        dict_cont_student,
        dict_discr_student,
        n_samples,
    )

    def test_student_mixture_attributes(self):
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
        SDEM = StudentDEMMD(
            is_dummy=False,
            type_discrete_features=type_qual_feats,
            index_discrete_features=index_qual_feats,
        )

        x_cont, x_discr = SDEM.transform_data(self.data, init=True)
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
        assert SDEM.eps == 1e-6  # default value for Student_DEM_MD
        assert SDEM.gamma == 1e-4  # default value for Student_DEM_MD

    def test_student_DEMMD_log_probabilities(self):

        data_cont = self.data[:, :2]

        log_prob_naive = np.empty((self.n_components, self.n_samples))
        for i, (mean, std, df, prop) in enumerate(
            zip(self.locations, self.scales, self.dofs, self.proportions)
        ):

            log_prob_naive[i, :] = stats.multivariate_t.logpdf(data_cont, mean, std, df) + np.log(
                prop
            )

        log_prob = estimate_student_log_proba(
            data_cont, self.locations, self.scales, self.dofs, self.proportions
        )

        assert_array_almost_equal(log_prob, log_prob_naive)

    def test_student_DEMMD_estimate_log_prob_resp(self):
        data_cont = self.data[:, :2]

        type_qual_feats = ["Poisson", "Multinomial", "Bernoulli"]
        index_qual_feats = [np.array([2]), np.array([3]), np.array([4])]

        SDEM = StudentDEMMD(
            is_dummy=False,
            type_discrete_features=type_qual_feats,
            index_discrete_features=index_qual_feats,
        )

        SDEM.fit(self.data)
        _, resp, _ = SDEM._e_step(self.data)  # n_components, n_samples
        assert_array_almost_equal(np.exp(resp).sum(axis=0), np.ones(self.n_samples))

    def test_student_mixture_fit_predict(self):
        import copy

        type_qual_feats = ["Poisson", "Multinomial", "Bernoulli"]
        index_qual_feats = [np.array([2]), np.array([3]), np.array([4])]

        SDEM = StudentDEMMD(
            is_dummy=False,
            type_discrete_features=type_qual_feats,
            index_discrete_features=index_qual_feats,
        )

        # check if fit_predict(X) is equivalent to fit(X).predict(X)
        f = copy.deepcopy(SDEM)
        Y_pred1 = f.fit(self.data).predict(self.data)
        Y_pred2 = SDEM.fit_predict(self.data)
        assert SDEM.n_components == self.n_components
        assert_array_equal(Y_pred1, Y_pred2)
