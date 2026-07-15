"""
File with simulation functions for gaussians mixture distributions
"""

from scipy.stats import multivariate_normal
import numpy as np


def simulation_mixed_data(cont_law, disc_law, cont_params, discr_params, n_samples):
    """With distribution names and true parameters,
    sample a dataset of size n_samples.

    Parameters
    ----------
    cont_law : string
            considered multivariate continuous distribution
    disc_law : list
            list of considered discrete distributions, in the same order than in discr_params
    cont_params : dict
            locations, scales, proportions, and additional parameterss
    discr_params : dict
            Discrete distributions's parameters. Keys are with the following format:  ["Bernoulli0","Poisson1",...]
    n_samples : integer
            wanted size of the dataset

    Returns
    -------
    array-like, (n_samples, n_features_cont + n_features_discr)
            dataset
    array-like, (n_samples,)
            hard assignments

    Raises
    ------
    ValueError
            Unknown distributions
    """

    n_components, n_features_cont = cont_params["locations"].shape

    labels = []
    obs = []
    disc_obs = []
    n_clusters = [
        int(np.round(n_samples * cont_params["proportions"][c])) for c in range(n_components)
    ]

    if sum(n_clusters) < n_samples:
        for i in range(n_samples - sum(n_clusters)):
            n_clusters[i] += 1
    elif sum(n_clusters) > n_samples:
        for i in range(sum(n_clusters) - n_samples):
            n_clusters[i] -= 1

    for c in range(n_components):

        if cont_law == "Gaussian":
            x = np.random.multivariate_normal(
                cont_params["locations"][c], cont_params["scales"][c], (n_samples,)
            )
            x = x[0 : n_clusters[c]]

        elif cont_law == "Student":
            y = np.random.chisquare(cont_params["dofs"][c], n_samples) / cont_params["dofs"][c]
            z = np.random.multivariate_normal(
                np.zeros(n_features_cont), cont_params["scales"][c], (n_samples,)
            )
            x = cont_params["locations"][c] + z / np.sqrt(y)[:, None]
            x = x[0 : n_clusters[c]]

        elif cont_law == "SAL":
            Y = np.random.multivariate_normal(
                np.zeros(n_features_cont), cont_params["scales"][c], (n_samples,)
            )
            W = np.random.exponential(scale=1.0, size=(n_samples, 1))
            x = cont_params["locations"][c] + W * cont_params["alphas"][c] + np.sqrt(W) * Y
            x = x[0 : n_clusters[c]]

        else:
            raise ValueError("Unimplemented distribution '%s'" % cont_law)

        obs.append(x)
        labels.append(np.repeat(c, n_clusters[c]))
        discr_array = []
        for j, (type_var) in enumerate(disc_law):
            if type_var != "None":
                if type_var == "Bernoulli":
                    var_disc = np.random.binomial(
                        1, discr_params[type_var + str(j)][c], (n_clusters[c], 1)
                    )
                    discr_array.append(var_disc)
                elif type_var == "Multinomial":
                    cat = np.random.multinomial(
                        1, discr_params[type_var + str(j)][c], size=(n_clusters[c], 1)
                    )
                    var_disc = np.argwhere(cat == 1)[:, 2] + 1
                    discr_array.append(var_disc.reshape(n_clusters[c], 1))
                elif type_var == "Poisson":
                    var_disc = np.random.poisson(
                        lam=discr_params[type_var + str(j)][c], size=(n_clusters[c], 1)
                    )
                    discr_array.append(var_disc)
                else:
                    raise ValueError("Unimplemented distribution '%s'" % type_var)

        if disc_law != ["None"]:
            disc_obs.append(np.array(discr_array).reshape(-1, n_clusters[c]).T)

    obs = np.vstack(obs)
    labels = np.hstack(labels)

    if disc_law != ["None"]:
        disc_obs = np.vstack(disc_obs)
        data = np.hstack([obs, disc_obs])
    else:
        data = obs

    if data.shape[0] != n_samples:
        print("Size of datasets does not correspond to n_samples number.")

    return data, labels
