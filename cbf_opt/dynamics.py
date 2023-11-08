import abc
import numpy as np
from typing import Tuple
from cbf_opt.tests import test_dynamics


class Dynamics(metaclass=abc.ABCMeta):
    def __init__(self, params: dict, test: bool = True, **kwargs):
        self.n_dims = len(self.STATES) if hasattr(self, "STATES") else kwargs["n_dims"]
        self.control_dims = len(self.CONTROLS) if hasattr(self, "CONTROLS") else kwargs["control_dims"]
        self.disturbance_dims = (
            len(self.DISTURBANCES) if hasattr(self, "DISTURBANCES") else kwargs.get("disturbance_dims", 0)
        )
        self.all_params = params
        # Make a new dictionary with the fixed parameters set to their values, and the uncertain parameters set to None
        self.params = {k: None if not isinstance(v, (int, float)) else v for k, v in params.items()}
        self.nbr_uncertain_params = len([v for v in self.params.values() if v is None])
        self.dt = kwargs["dt"]
        self.step_scheme = kwargs.get("step_scheme", "fe")
        self.periodic_dims = self.PERIODIC_DIMS if hasattr(self, "PERIODIC_DIMS") else []
        if test:
            test_dynamics.test_dynamics(self)

    @abc.abstractmethod
    def f(self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray = None, time: float = 0.0):
        """Implements the continuous-time dynamics ODE: dx_dt = f(x, t)"""

    def __call__(
        self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray = None, time: float = 0.0
    ) -> np.ndarray:
        # assert all params are floats or ints
        assert all(
            [isinstance(v, (float, int)) for v in self.params.values()]
        ), "All params must be fixed for simulation"
        return self.f(state, control, disturbance, time)

    def wrap_dynamics(self, state: np.ndarray) -> np.ndarray:
        """
        Periodic dimensions are wrapped to [-pi, pi)

        Args:
            state (np.ndarray): Unwrapped state

        Returns:
            state (np.ndarray): Wrapped state
        """
        for periodic_dim in self.periodic_dims:
            try:
                state[..., periodic_dim] = (state[..., periodic_dim] + np.pi) % (2 * np.pi) - np.pi
            except TypeError:  # FIXME: Clunky at best, how to deal with jnp and np mix
                state = state.at[periodic_dim].set((state[periodic_dim] + np.pi) % (2 * np.pi) - np.pi)

        return state

    def state_jacobian(
        self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray = None, time: float = 0.0
    ) -> np.ndarray:
        raise NotImplementedError("Define state_jacobian in subclass")

    @abc.abstractmethod
    def control_jacobian(
        self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray = None, time: float = 0.0
    ) -> np.ndarray:
        """Implements the control Jacobian df(x,u,d,t) / du"""

    @abc.abstractmethod
    def disturbance_jacobian(
        self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray = None, time: float = 0.0
    ) -> np.ndarray:
        raise NotImplementedError("Define disturbance_jacobian in subclass")

    def linearized_ct_dynamics(
        self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray = None, time: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the linearized state and control matrices, A, B and W of the continuous-time dynamics"""
        return (
            self.state_jacobian(state, control, disturbance, time),
            self.control_jacobian(state, control, disturbance, time),
            self.disturbance_jacobian(state, control, disturbance, time),
        )

    def linearized_dt_dynamics(
        self,
        state: np.ndarray,
        control: np.ndarray,
        disturbance: np.ndarray = None,
        time: float = 0.0,
        dt: float = None,
    ):
        """Implements the linearized discrete-time dynamics"""
        dt = self.dt if dt is None else dt
        A, B, D = self.linearized_ct_dynamics(state, control, disturbance, time)
        A_d = np.eye(self.n_dims) + dt * A
        B_d = dt * B
        D_d = dt * D
        return A_d, B_d, D_d

    def step(
        self,
        state: np.ndarray,
        control: np.ndarray,
        disturbance: np.ndarray = None,
        time: float = 0.0,
        dt: float = None,
    ) -> np.ndarray:
        """Implements the discrete-time dynamics ODE
        scheme in {fe, rk4}"""
        dt = self.dt if dt is None else dt
        if self.step_scheme == "fe":
            n_state = state + self(state, control, disturbance, time) * dt
        elif self.step_scheme == "rk4":
            # TODO: Figure out how to do RK4 with periodic dimensions (aka angle normalization)
            # Assumes zoh on control
            k1 = self(state, control, disturbance, time)
            k2 = self(state + k1 * dt / 2, control, disturbance, time + dt / 2)
            k3 = self(state + k2 * dt / 2, control, disturbance, time + dt / 2)
            k4 = self(state + k3 * dt, control, disturbance, time + dt)
            n_state = state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        else:
            raise ValueError("scheme must be either 'fe' or 'rk4'")
        return self.wrap_dynamics(n_state)


class ControlAffineDynamics(Dynamics):
    def __init__(self, params: dict, test: bool = True, **kwargs):
        super().__init__(params, test, **kwargs)
        if test:
            test_dynamics.test_control_affine_dynamics(self)

    def f(
        self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray = None, time: float = 0.0
    ) -> np.ndarray:
        """Implements the continuous-time dynamics ODE: dx_dt = f(x, t) + g(x, t) @ u"""
        disturbance = np.zeros((self.disturbance_dims,)) if disturbance is None else disturbance
        return (
            self.open_loop_dynamics(state, time)
            + self.control_matrix(state, time) @ control
            + self.disturbance_jacobian(state, time) @ disturbance
        )

    @abc.abstractmethod
    def open_loop_dynamics(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the open loop dynamics f(x,t)"""

    def control_jacobian(
        self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray = None, time: float = 0.0
    ) -> np.ndarray:
        """For control affine systems the control_jacobian is equivalent to the control matrix"""
        return self.control_matrix(state, time)

    @abc.abstractmethod
    def control_matrix(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the control Jacobian g(x,u) f(x,u,t)"""

    def disturbance_jacobian(
        self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray = None, time: float = 0.0
    ) -> np.ndarray:
        return self.disturbance_matrix(state, time)

    def disturbance_matrix(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the disturbance Jacobian w(x,u,t)"""
        return np.zeros((self.n_dims, self.disturbance_dims))


class BatchedDynamics:
    def __init__(self, dyn):
        import jax

        self.dyn = dyn
        for attr_name, attr_value in vars(self.dyn).items():
            setattr(self, attr_name, attr_value)

        for attr_name, attr_value in vars(self.dyn.__class__).items():
            if not callable(attr_value) and not attr_name.startswith("__"):
                setattr(self, attr_name, attr_value)

        self.control_jacobian = jax.vmap(dyn.control_jacobian)
        self.disturbance_jacobian = jax.vmap(dyn.disturbance_jacobian)
        self.state_jacobian = jax.vmap(dyn.state_jacobian)
        self.step = jax.vmap(dyn.step)

        if hasattr(dyn, "open_loop_dynamics"):
            self.open_loop_dynamics = jax.vmap(dyn.open_loop_dynamics)
        if hasattr(dyn, "control_matrix"):
            self.control_matrix = jax.vmap(dyn.control_matrix)
        if hasattr(dyn, "disturbance_matrix"):
            self.disturbance_matrix = jax.vmap(dyn.disturbance_matrix)

    @property
    def params(self):
        return self.dyn.params

    @params.setter
    def params(self, new_value):
        self.dyn.params = new_value


# TODO: Build wrapper for interface with openai gym (here the state is a class instance variable)
class PartialObservableDynamics(Dynamics):
    def __init__(self, params: dict, test: bool = True, **kwargs):
        super().__init__(params, test, **kwargs)
        self.obs_dims = params["obs_dims"]

    @abc.abstractmethod
    def observe(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the observation function h(x,t)"""
