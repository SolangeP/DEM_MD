# Dynamic Expectation-Maximization algorithms for Mixed-type Data

This repository provides tools and algorithms for the estimation of mixture models for mixed-type data. The algorithms **jointly** estimate the model parameters **and** the number of classes in the model.

This repository corresponds to the implementation of the paper 📄  *[Solange Pruilh, Stéphanie Allassonnière. Dynamic Expectation-Maximization algorithms for Mixed-type Data. 2024.](https://hal.science/hal-04510689)*.


## Requirements

The code was tested on Python 3.12.14 . In order to run the code, the Python packages listed in `requirements.txt` are needed. They can be installed for instance with conda.

```python
conda create -n DEM_MD --file requirements.txt
conda activate DEM_MD
```

## Demonstration 

A jupyter notebook provides detailed experiments on different use cases: [`Tutorial on DEM-MD algorithms`](Tutorial on DEM-MD algorithms.ipynb). 

## Code

Several classes are provided, corresponding to DEM-MD algorithms to estimate different mixture models.

- `DEM_MD_gaussian.py`, `DEM_MD_student.py` and `DEM_MD_sal.py` contains classes to estimate mixture models with respectively Gaussian, Student and Shifted Asymmetric Laplace distributions.
- These three DEM-MD classes are based on `base_DEM_MD.py`, `base_MD.py` and `base.py` which are parents classes. 
- `utils` folder contains several files with tool functions, `sampling.py` contains a function to sample datasets, `calculation.py` contains miscellaneous functions.
- `history.py` contains `Historic` class, which is directly instantiated into DEM-MD classes.