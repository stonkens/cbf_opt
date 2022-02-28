import abc
import numpy as np


class Dynamics(metaclass=abc.ABCMeta):
    def __init__(self, params: dict, **kwargs):
        self.n_dims = params["n_dims"]
        self.control_dims = params["control_dims"]
        self.disturbance_dims = params.get("disturbance_dims", 1)
        self.dt = params["dt"]

    @abc.abstractmethod
    def __call__(self, state: np.ndarray, control: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the continuous-time dynamics ODE"""

    def state_jacobian(
        self, state: np.ndarray, control: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        raise NotImplementedError("Define state_jacobian in subclass")

    def control_jacobian(
        self, state: np.ndarray, control: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        raise NotImplementedError("Define control_jacobian in subclass")

    def linearized_ct_dynamics(self, state: np.ndarray, control: np.ndarray, time: float = 0.0):
        return self.state_jacobian(state, control, time), self.control_jacobian(
            state, control, time
        )

    def linearized_dt_dynamics(self, state: np.ndarray, control: np.ndarray, time: float = 0.0):
        """Implements the linearized discrete-time dynamics"""
        A, B = self.linearized_ct_dynamics(state, control, time)
        A_d = np.eye(self.n_dims) + self.dt * A
        B_d = self.dt * B
        return A_d, B_d

    def step(self, state: np.ndarray, control: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the discrete-time dynamics ODE"""
        # TODO: Add compatibility with more complicated interpolation schemes
        return state + self(state, control, time) * self.dt


class ControlAffineDynamics(Dynamics):
    def __init__(self, params: dict, **kwargs):
        super().__init__(params, **kwargs)
        assert self.open_loop_dynamics(np.random.rand(self.n_dims)).shape[0] == self.n_dims
        assert self.control_matrix(np.random.rand(self.n_dims)).shape == (
            self.n_dims,
            self.control_dims,
        )

    def __call__(self, state: np.ndarray, control: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the continuous-time dynamics ODE: dx_dt = f(x, t) + g(x, t) @ u"""
        return self.open_loop_dynamics(state, time) + self.control_matrix(state, time) @ control

    @abc.abstractmethod
    def open_loop_dynamics(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the open loop dynamics f(x,t)"""

    def control_jacobian(
        self, state: np.ndarray, control: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        return self.control_matrix(state, time)

    @abc.abstractmethod
    def control_matrix(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the control Jacobian g(x,u) f(x,u,t)"""

    def disturbance_jacobian(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        return np.atleast_2d(np.zeros((self.n_dims, self.disturbance_dims)))
