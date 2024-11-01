# Notations guide

## Variables

- `x` is the array with all datapoints, of size `(n_samples,n_features)` with `n_features = n_features_cont + n_features_discr` with `n_features_cont` the number of features following an unique multivariate continuous distribution and `n_features_discr` the number of features following a discrete distribution.
- `x_cont` is a subarray from `x`, with only features distributed accoridng to a multivariate continuous distribution.
- `x_discr` is a subarray from `x`, with only features distributed accoridng to a discrete distribution.
- `tau` is the array of responsibilities, which are posterior probabilities computed at E-step. `tau` is of shape `(n_components, n_samples)`. `log_tau` is the corresponding array with logarithms of responsibilities.
- `locations` is the array of locations (centers, means for Gaussian or Student distribution for example) parameters. Shape `(n_components,n_features_cont)`
- `scales` is the array of scale matrices (covariances). Shape `(n_components, n_features_cont,n_features_cont)`
- `proportions` is the vector with mixture proportions, which are weights of each distribution in the mixture. Shape `(n_components,)` 
- `p_discrete` is the list of size `n_features_discr` with parameters for each discrete distribution. Each element in `p_discrete` is an array of shape `(n_components, n_modalities)` for Multinomial distribution, `(n_components, 1)` for Poisson or Bernoulli distributions.

## Variables per distribution
### Student mixture model
- `dofs` is a vector with degrees of freedom for the Student distributions. Shape `(n_components,)`
- `gamma_u`  are posterior probabilities of latent variables U in Student mixture models. Shape `(n_components,n_samples)`

### Shifted Asymmetric Laplace mixture model
- `alphas` is an array with skewness parameters for the Shifted Asymmetric Laplace (SAL) distributions. Shape `(n_components,n_features_cont)`
- `e1` is an array with expected value of latent variables W in SAL mixture models. Shape `(n_components,n_samples)`
- `e2` is an array with expected value of latent variables W^-1 in SAL mixture models. Shape `(n_components,n_samples)`

## Historic object

- `Historic` object is an inherent object of each DEM-MD class. 
- In an `Historic` object: variables  of a mixture class are saved as elements of lists. And these lists are incremented over iterations.
- For example, `Historic.n_components` is a list where elements are the number of components at each iteration of the DEM-MD estimation process. `Historic.means` is a list where elements are arrays of estimated means at each iteration of the DEM-MD estimation process.
- With `export_history` method of a DEM-MD class, its variables and attributes are exported as a dictionnary, including variables stored in an Historic object.