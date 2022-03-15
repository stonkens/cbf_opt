# Control Barrier Function toolbox
Toolbox for implementing safety filters using Control Barrier Functions (CBFs) in `python`. Implemented functionality for CBFs and implicit CBFs that rely on a backup controller.

Implementation of the solver is provided for control-affine systems (with bounded input constraints) using `cvxpy`. 

Defines Experiment class to efficiently run experiments with different (or none) CBFs

The toolbox is compatible with batched inputs (`torch` or `tf`) and individual inputs (`numpy` or `jax`)

## Installation:
- Run `pip install -e .` to install this project

## Requirements
- `cvxpy`

## TODOs
Testing CBF:
- Create a mock system which you can test some functionality on
    - V, gradV and V at next point to test delta_V (V + dx * gradV) vs (V_next - V)
    - Test Lie Derivatives: Compute V_dot with Lf_V and Lg_V and compare with tiny forward simulation
DataModule:
- Assert sizes are right 
Experiments:
- Have a MockExperiment class
- Assert that the results should be a list of dataframes (for the different experiments or test cases you run). 
- Assert that there are multiple figures etc. (see https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/experiments/tests/test_experiment_suite.py)

Testing any random system:
- Assert the nominal control is of the right size
- Assert the things we need for the backup controller, sometimes also just the type

Do we want to have it work for batched inputs?
What about the jnp vs np integration --> What do we propose as solution here?
