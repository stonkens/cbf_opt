import abc
import numpy as np
from typing import Tuple
from cbf_opt.tests import test_dynamics


class Dynamics(metaclass=abc.ABCMeta):
    def __init__(self, params: dict, test: bool = True, **kwargs):
        self.n_dims = params["n_dims"]
        self.control_dims = params["control_dims"]
        self.disturbance_dims = params.get("disturbance_dims", 1)
        self.dt = params["dt"]
        if test:
            test_dynamics.test_dynamics(self)

    @abc.abstractmethod
    def __call__(self, state: np.ndarray, control: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the continuous-time dynamics ODE \dot{x} = f(x, u, t)"""

    def state_jacobian(
        self, state: np.ndarray, control: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        raise NotImplementedError("Define state_jacobian in subclass")

    def control_jacobian(
        self, state: np.ndarray, control: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        raise NotImplementedError("Define control_jacobian in subclass")

    def linearized_ct_dynamics(
        self, state: np.ndarray, control: np.ndarray, time: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the linearized state and control matrices, A and B, of the continuous-time dynamics"""
        return self.state_jacobian(state, control, time), self.control_jacobian(
            state, control, time
        )

    def linearized_dt_dynamics(self, state: np.ndarray, control: np.ndarray, time: float = 0.0):
        """Implements the linearized discrete-time dynamics"""
        A, B = self.linearized_ct_dynamics(state, control, time)
        A_d = np.eye(self.n_dims) + self.dt * A
        B_d = self.dt * B
        return A_d, B_d

    def step(
        self, state: np.ndarray, control: np.ndarray, time: float = 0.0, scheme: str = "fe"
    ) -> np.ndarray:
        """Implements the discrete-time dynamics ODE
        scheme in {fe, rk4}"""
        if scheme == "fe":
            return state + self(state, control, time) * self.dt
        elif scheme == "rk4":
            # Assumes zoh on control
            k1 = self(state, control, time)
            k2 = self(state + k1 * self.dt / 2, control, time + self.dt / 2)
            k3 = self(state + k2 * self.dt / 2, control, time + self.dt / 2)
            k4 = self(state + k3 * self.dt, control, time + self.dt)
            return state + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
        else:
            raise ValueError("scheme must be either 'fe' or 'rk4'")

    # TODO: Add angle normalization after computing dynamics (think carefully where to add!)


class ControlAffineDynamics(Dynamics):
    def __init__(self, params: dict, test: bool = True, **kwargs):
        super().__init__(params, test, **kwargs)
        if test:
            test_dynamics.test_control_affine_dynamics(self)

    def __call__(self, state: np.ndarray, control: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the continuous-time dynamics ODE: dx_dt = f(x, t) + g(x, t) @ u"""
        return (
            self.open_loop_dynamics(state, time)
            + (self.control_matrix(state, time) @ np.atleast_3d(control)).squeeze()
        )

    @abc.abstractmethod
    def open_loop_dynamics(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the open loop dynamics f(x,t)"""

    def control_jacobian(
        self, state: np.ndarray, control: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """For control affine systems the control_jacobian is equivalent to the control matrix"""
        return self.control_matrix(state, time)

    @abc.abstractmethod
    def control_matrix(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the control Jacobian g(x,u) f(x,u,t)"""

    def disturbance_jacobian(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        # TODO: Is this required?
        return np.atleast_2d(np.zeros((self.n_dims, self.disturbance_dims)))
