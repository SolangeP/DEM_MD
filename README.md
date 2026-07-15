# Dynamic Expectation-Maximization algorithms for Mixed-type Data

This repository provides tools and algorithms for the estimation of mixture models for mixed-type data. The algorithms **jointly** estimate the model parameters **and** the number of classes in the model.

This repository corresponds to the implementation of the paper 📄  *[Solange Pruilh, Stéphanie Allassonnière. Dynamic Expectation-Maximization algorithms for Mixed-type Data. 2024.](https://hal.science/hal-04510689)*.


## Requirements

The code was tested on Python 3.12.4.

**With uv** (recommended — installs Python and all dependencies in one step):

```bash
pip install uv          # install uv once, globally
uv sync                 # creates .venv/ and installs all dependencies
uv run python           # run any command inside the environment
```

**With conda** (alternative):

```bash
conda create -n DEM_MD --file requirements.txt
conda activate DEM_MD
pip install -e .
```

## Quick start

```python
import numpy as np
from dem_md import GaussianDEMMD

# Continuous-only data
X = np.random.randn(200, 4)
model = GaussianDEMMD()
labels = model.fit_predict(X)

# Mixed data (continuous + one Bernoulli discrete feature)
X_mixed = np.hstack([np.random.randn(200, 3), np.random.randint(0, 2, (200, 1))])
model_mixed = GaussianDEMMD(
    type_discrete_features=["Bernoulli"],
    index_discrete_features=[np.array([3])],
)
labels_mixed = model_mixed.fit_predict(X_mixed)
```

See the [tutorial notebook](<Tutorial on DEM-MD algorithms.ipynb>) for detailed experiments on different use cases.

## Code

Several classes are provided, corresponding to DEM-MD algorithms to estimate different mixture models.

- `DEM_MD_gaussian.py`, `DEM_MD_student.py` and `DEM_MD_sal.py` contain classes to estimate mixture models with respectively Gaussian (`GaussianDEMMD`), Student (`StudentDEMMD`) and Shifted Asymmetric Laplace (`SALDEMMD`) distributions.
- These three DEM-MD classes are based on `base_DEM_MD.py`, `base_MD.py` and `base.py` which are parent classes.
- `utils` folder contains several files with tool functions: `sampling.py` for sampling datasets, `calculation.py` for miscellaneous computations, and `validation.py` for input validation.
- `history.py` contains the `Historic` class, which is directly instantiated into DEM-MD classes.
- `Notations.md` documents the variable naming conventions used throughout the codebase.

## Citation

```bibtex
@unpublished{pruilh2024demmd,
  title     = {Dynamic Expectation-Maximization algorithms for Mixed-type Data},
  author    = {Pruilh, Solange and Allassonni{\`e}re, St{\'e}phanie},
  year      = {2024},
  note      = {Preprint},
  url       = {https://hal.science/hal-04510689},
}
```
