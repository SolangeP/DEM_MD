import numbers
import numpy as np
from scipy.spatial.distance import cdist

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f'{seed} cannot be used to seed a numpy.random.RandomState instance')


def check_patho_clusters(locations, scales):
    """Method to check if some clusters are superimposed with the following rule: identical locations and identical scales.

    Returns
    -------
    patho : bool
        Boolean giving state of superimposed clusters
    """
    
    n_components, _ = locations.shape
    patho = False
    dist_centroids = cdist(locations, locations, metric="euclidean")
    dist_cov = np.zeros((n_components, n_components))

    for i in range(n_components):
        for j in range(i, n_components):
            dist_cov[i, j] = np.linalg.norm(
                (scales[i] - scales[j])
            )
            dist_cov[j, i] = np.linalg.norm(
                (scales[j] - scales[i])
            )

    dist_totale = dist_cov + dist_centroids
    index_null = np.argwhere(dist_totale == 0)

    if len(index_null) > n_components:
        patho = True

    return patho
