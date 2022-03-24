# Control Barrier Function toolbox
Toolbox for implementing safety filters using Control Barrier Functions (CBFs) in `python`. Implemented functionality for CBFs and implicit CBFs that rely on a backup controller.

Implementation of the solver is provided for control-affine systems (with bounded input constraints) using `cvxpy`. 

Defines Experiment class to efficiently run experiments with different (or none) CBFs

The toolbox is compatible with batched inputs (`torch` or `tf`) and individual inputs (`numpy` or `jax`)

## Installation:
- Run `pip install -e .` to install this project

Example available in `examples/acc.ipynb`
