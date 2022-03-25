# Control Barrier Function toolbox
Toolbox for implementing safety filters using Control Barrier Functions (CBFs) in `python` using `cvxpy`. 

[Control barrier functions](https://arxiv.org/abs/1903.11199) are a principled tool to encode (through a scalar value function) and enforce (through a condition on the derivative of this value function) safety properties of a system. Enforcing safety using CBFs is typically implemented in a safety filter (also referred to as active safety invariance filter, ASIF) by minimally modifiying a nominal (safety-agnostic) policy to maintain safety. For control-affine systems, $\dot x = f(x) + g(x)u$, the CBF condition is linear in the control $u$ and can be efficiently solved online. This package provides an implementation of this using `cvxpy`. The solved constraint is then $u^* = \arg \min_u \{\lVert u - u_{\text{nom}} \rVert_2^2 \mid \dot h + \alpha h \geq 0\}$, with $h$ the CBF and $\alpha$ the maximal safety decay rate.

## Installation
- Run `pip install -e .` to install this project and its dependencies (from `requirements.txt`).

## Instructions
`dynamics.py` provides abstract classes for different types of dynamics, `cbf.py` for different type of CBFs, and `asif.py` for different types of safety filters. The toolbox is compatible with batched inputs (`torch`, `tf` or `numpy`) and individual inputs (`numpy` or `jax`)

To use this toolbox, a user defines the dynamics for their problem, the cbf, and the safety filter (solely requires setting $\alpha$ and the nominal policy). The user then can run experiments manually or using the `experiment-wrapper` toolbox [link](https://github.com/stonkens/experiment_wrapper). Both use-cases are showcased in the examples folder.

An example for the control-affine setting, the Adaptive Cruise Control problem, is available in `examples/acc.ipynb`.
