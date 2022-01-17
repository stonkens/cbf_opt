import abc
import cvxpy as cp
import numpy as np


class ASIF(metaclass=abc.ABCMeta):
    def __init__(self, dynamics, cbf, **kwargs) -> None:
        self.dynamics = dynamics
        self.cbf = cbf
        self.alpha = kwargs.get("alpha", lambda x: x)
        self.nominal_policy = kwargs.get(
            "nominal_policy", lambda x, t: 0
        )  # TODO: Implement correct dims for control output

    @abc.abstractmethod
    def __call__(self, state, nominal_control, time):
        """Implements the active safety invariance filter"""


class ControlAffineASIF(ASIF):
    def __init__(self, dynamics, cbf, **kwargs) -> None:
        super().__init__(dynamics, cbf, **kwargs)
        self.filtered_control = cp.Variable(self.dynamics.control_dims)
        self.nominal_control = cp.Parameter(self.dynamics.control_dims)
        self.b = cp.Parameter(1)
        self.A = cp.Parameter((1, self.dynamics.control_dims))
        """
        min || u - u_des ||^2
        s.t. A @ u + b >= 0
        """
        self.obj = cp.Minimize(cp.norm(self.filtered_control - self.nominal_control, 2) ** 2)
        self.constraints = [self.A @ self.filtered_control + self.b >= 0]
        self.QP = cp.Problem(self.obj, self.constraints)

    def __call__(self, state, nominal_control=None, time=0.0):
        h = self.cbf.vf(state, time)
        Lf_h, Lg_h = self.cbf.lie_derivatives(state, time)
        self.b.value = self.alpha(h) + Lf_h
        self.A.value = Lg_h

        if nominal_control is not None:
            assert nominal_control.shape == (self.dynamics.control_dims,)
            self.nominal_control.value = nominal_control
        else:
            self.nominal_control.value = self.nominal_policy(state, time)

        self.QP.solve()
        # TODO: Check if the solution is feasible
        return np.atleast_1d(self.filtered_control.value)
