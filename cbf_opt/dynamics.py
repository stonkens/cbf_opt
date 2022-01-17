import abc


class Dynamics(metaclass=abc.ABCMeta):
    def __init__(self, params, *args, **kwargs):
        self.n_dims = params["n_dims"]
        self.control_dims = params["control_dims"]
        self.dt = params["dt"]

    @abc.abstractmethod
    def __call__(self, state, control, time=None):
        """Implements the continuous-time dynamics ODE"""

    def step(self, state, control, time=None):
        """Implements the discrete-time dynamics ODE"""
        return state + self(state, control, time) * self.dt


class ControlAffineDynamics(Dynamics):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        # TODO: Assert dimensions of open_loop_dynamics and control_jacobian

    def __call__(self, state, control, time=None):
        """Implements the continuous-time dynamics ODE: dx_dt = f(x, t) + g(x, t) @ u"""
        return self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control

    @abc.abstractmethod
    def open_loop_dynamics(self, state, time=None):
        """Implements the open loop dynamics f(x,t)"""

    @abc.abstractmethod
    def control_jacobian(self, state, time=None):
        """Implements the control Jacobian g(x,u) f(x,u,t)"""
