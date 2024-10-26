"""
File with calculus functions
"""

from itertools import permutations
import numpy as np
import scipy.spatial as sp

############# Methods for covariance computations


def _estimate_gaussian_covariances_full(tau, x, nk, means, reg_covar):
    """
    Compute covariance parameters in multivariate Gaussian distributions

    Parameters
    ----------
    tau : see Notations.md
    x : see Notations.md
    nk: Sum of responsibilities for each component (over samples)
    means : see Notations.md
    reg_covar : float
        value to regularize covariance matrices

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The scale matrices of the current components.

    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))

    for k in range(n_components):
        diff = x - means[k]
        covariances[k] = np.dot(tau[k, :] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar

    return covariances


def _estimate_student_covariances_full(tau, gamma_u, x, means, reg_covar):
    """Compute covariance parameters in multivariate Student distributions

    Parameters
    ----------
    tau : see Notations.md
    gamma_u : see Notattions.md
    x : see Notations.md
    means : see Notations.md
    reg_covar : float
        value to regularize covariance matrices


    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The scale matrices of the current components.
    """

    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))

    for k in range(n_components):
        diff = x - means[k]
        covariances[k] = np.dot(tau[k, :] * gamma_u[k, :] * diff.T, diff) / (tau[k, :]).sum()
        covariances[k].flat[:: n_features + 1] += reg_covar

    return covariances


############# Methods for permutations and errors on parameters


def compute_min_permut(est_means, true_means):
    """Find optimal matching between estimated and true classes, based on location parameters.

    Parameters
    ----------
    est_means : array-like, (n_components, n_features)
        Array with estimated location parameters
    true_means : array-like, (n_components, n_features)
        Array with true location parameters

    Returns
    -------
    list
        Indexes to match estimated classes with true classes
    """

    true_k = true_means.shape[0]

    dist_matrix = sp.distance_matrix(true_means, est_means)

    l = list(permutations(np.arange(0, true_k)))
    min_val = np.inf
    min_permutation = l[0]

    for permut in l:
        val = []
        for k in range(true_k):
            val.append(dist_matrix[k, permut[k]])

        if sum(val) <= min_val:
            min_val = sum(val)
            min_permutation = permut
        del val

    min_permutation = list(min_permutation)
    return min_permutation


def swap_labels(est_means, labels, dico_cont_params, permut):
    """Swap labels in estimated_labels vect
    to match classes with the true mixture model

    Parameters
    ----------
    est_means : array-like, (n_components, n_features)
        estimated means
    labels : (n_samples,)
        estimated labels (hard assignment)
    dico_cont_params : dict
        dictionnary with true parameters
    permut : list,
        Indexes to match estimated classes with true ones

    Returns
    -------
    list
        Permuted estimated labels to match estimated classes with true ones
    """

    est_k = est_means.shape[0]

    true_k = len(dico_cont_params["proportions"])

    permut_labels = np.zeros(labels.shape, dtype=int)

    if est_k == true_k:
        for j in range(true_k):
            mask = labels == permut[j]
            permut_labels[mask] = j

    else:
        permut_labels = labels.copy()
    return permut_labels


def errors_proportions(est_pi, true_pi, permutation):
    """Compute relative errors of estimated parameters.

    Parameters
    ----------
    est_pi : array-like, shape (n_components, )
        Array of estimated proportion parameters
    true_pi : array-like, shape (n_components, )
        Array of true proportion parameters
    permutation : list
        list of indexes to match estimated and true classes

    Returns
    -------
    array-like, (n_components, )
        relative errors for each class
    """
    true_k = len(true_pi)

    return np.array(
        [np.abs(est_pi[permutation[k]] - true_pi[k]) / np.abs(true_pi[k]) for k in range(true_k)]
    )


def errors_locations(est_means, true_means, permutation):
    """Compute relative errors of estimated parameters.

    Parameters
    ----------
    est_means : array-like, shape (n_components, n_features_cont)
        Array of estimated locations (means) parameters
    true_means : array-like, shape (n_components, n_features_cont)
        Array of true locations (means) parameters
    permutation : list
        list of indexes to match estimated and true classes

    Returns
    -------
    array-like (n_components,)
        relative errors for each class
    """
    true_k = true_means.shape[0]

    errors_means = [
        np.linalg.norm(est_means[permutation[k]] - true_means[k]) / np.linalg.norm(true_means[k])
        for k in range(true_k)
    ]
    return np.array(errors_means)


def errors_scales(est_cov, true_cov, permutation):
    """Compute relative errors of estimated parameters.

    Parameters
    ----------
    est_cov : array-like, shape (n_components, n_features_cont, n_features_cont)
        Array of estimated scales (covariances) parameters
    true_cov : array-like, shape (n_components, n_features_cont, n_features_cont)
        Array of true scales (covariances) parameters
    permutation : list
        list of indexes to match estimated and true classes

    Returns
    -------
    array-like, (n_components,)
        relative errors for each class
    """

    true_k = true_cov.shape[0]

    Frob_norm = [
        np.linalg.norm(est_cov[permutation[j]] - true_cov[j]) / np.linalg.norm(true_cov[j])
        for j in range(true_k)
    ]

    return np.array(Frob_norm)


def errors_dofs(est_dof, true_dof, permutation):
    """Compute relative errors of estimated parameters.

    Parameters
    ----------
    est_dof : array-like, shape (n_components, )
        Array of estimated d.o.f parameters
    true_dof : array-like, shape (n_components, )
        Array of true d.o.f parameters
    permutation : list
        list of indexes to match estimated and true classes

    Returns
    -------
    array-like, (n_components, )
        relative errors for each class
    """
    true_k = len(true_dof)

    return np.array(
        [np.abs(true_dof[k] - est_dof[permutation[k]]) / np.abs(true_dof[k]) for k in range(true_k)]
    )


def errors_skeweness(est_alpha, true_alpha, permutation):
    """Compute relative errors of estimated parameters.

    Parameters
    ----------
    est_alpha : array-like, shape (n_components, n_features_cont)
        Array of estimated skewness parameters
    true_alpha : array-like, shape (n_components, n_features_cont)
        Array of true skewness parameters
    permutation : list
        list of indexes to match estimated and true classes

    Returns
    -------
    array-like, (n_components, )
        relative errors for each class
    """

    true_k = true_alpha.shape[0]

    return np.array(
        [
            np.linalg.norm(true_alpha[k] - est_alpha[permutation[k]], 2)
            / np.linalg.norm(true_alpha[k])
            for k in range(true_k)
        ]
    )


def errors_bernoulli(est_ber, true_ber, permutation):
    """Compute relative errors of estimated parameters.

    Parameters
    ----------
    est_ber : array-like, shape (n_components, )
        Array of estimated bernoulli distribution parameters
    true_ber : array-like, shape (n_components, )
        Array of true bernoulli distribution parameters
    permutation : list
        list of indexes to match estimated and true classes

    Returns
    -------
    array-like, (n_components, )
        relative errors for each class
    """
    true_k = len(true_ber)

    return np.array(
        [np.abs(est_ber[permutation[k]] - true_ber[k]) / np.abs(true_ber[k]) for k in range(true_k)]
    )


def errors_poisson(est_poisson, true_poisson, permutation):
    """Compute relative errors of estimated parameters.

    Parameters
    ----------
    est_poisson : array-like, shape (n_components, )
        Array of estimated poisson distribution parameters
    true_poisson : array-like, shape (n_components, )
        Array of true poisson distribution parameters
    permutation : list
        list of indexes to match estimated and true classes

    Returns
    -------
    array-like, (n_components, )
        _description_
    """
    true_k = len(true_poisson)

    return np.array(
        [
            np.abs(est_poisson[permutation[k]] - true_poisson[k]) / np.abs(true_poisson[k])
            for k in range(true_k)
        ]
    )


def errors_multi(est_multi, true_multi, permutation):
    """Compute relative errors of estimated parameters.

    Parameters
    ----------
    est_multi : array-like, shape (n_components, M)
        Array of estimated multinomial distribution parameters
    true_multi : array-like, shape (n_components, M)
        Array of true multinomial distribution parameters
    permutation : list
        list of indexes to match estimated and true classes

    Returns
    -------
    array-like, shape (n_components,)
        relative errors for each class
    """
    true_k = true_multi.shape[0]

    return np.array(
        [
            np.linalg.norm(est_multi[permutation[k]] - true_multi[k], ord=1)
            / np.linalg.norm(true_multi[k], ord=1)
            for k in range(true_k)
        ]
    )
