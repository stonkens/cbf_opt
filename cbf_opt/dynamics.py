import abc
import numpy as np


class Dynamics(metaclass=abc.ABCMeta):
    def __init__(self, params, *args, **kwargs):
        self.n_dims = params["n_dims"]
        self.control_dims = params["control_dims"]
        self.disturbance_dims = params.get("disturbance_dims", 1)
        self.dt = params["dt"]

    @abc.abstractmethod
    def __call__(self, state, control, time=None):
        """Implements the continuous-time dynamics ODE"""

    def state_jacobian(self, state, control, time=None):
        raise NotImplementedError("Define state_jacobian in subclass")

    def control_jacobian(self, state, control, time=None):
        raise NotImplementedError("Define control_jacobian in subclass")

    def step(self, state, control, time=None):
        """Implements the discrete-time dynamics ODE"""
        # TODO: Add compatibility with more complicated interpolation schemes
        return state + self(state, control, time) * self.dt


class ControlAffineDynamics(Dynamics):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        assert self.open_loop_dynamics(np.random.rand(self.n_dims)).shape[0] == self.n_dims
        assert self.control_matrix(np.random.rand(self.n_dims)).shape == (
            self.n_dims,
            self.control_dims,
        )

    def __call__(self, state, control, time=None):
        """Implements the continuous-time dynamics ODE: dx_dt = f(x, t) + g(x, t) @ u"""
        return self.open_loop_dynamics(state, time) + self.control_matrix(state, time) @ control

    @abc.abstractmethod
    def open_loop_dynamics(self, state, time=None) -> np.ndarray:
        """Implements the open loop dynamics f(x,t)"""

    def control_jacobian(self, state, control, time=None):
        return self.control_matrix(state, time)

    @abc.abstractmethod
    def control_matrix(self, state, time=None) -> np.ndarray:
        """Implements the control Jacobian g(x,u) f(x,u,t)"""

    def disturbance_jacobian(self, state, time=None):
        return np.atleast_2d(np.zeros((self.n_dims, self.disturbance_dims)))
