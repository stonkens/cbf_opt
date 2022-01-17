import abc
import numpy as np


class CBF(metaclass=abc.ABCMeta):
    def __init__(self, dynamics, params, **kwargs) -> None:
        self.dynamics = dynamics

    def is_safe(self, state, time):
        """
        :param state: state
        :param time: time
        :return: is safe
        """
        return self.vf(state, time) >= 0

    @abc.abstractmethod
    def vf(self, state, time) -> float:
        """Implements the value function h(x)"""

    def vf_dt(self, state, control, time):
        return self.vf_dt_partial(state, time) + self._grad_vf(state, time) @ self.dynamics(
            state, control, time
        )

    @abc.abstractmethod
    def vf_dt_partial(self, state, time):
        """Implements the partial derivative of the time derivative of the value function h(x)"""

    @abc.abstractmethod
    def _grad_vf(self, state, time):
        """Implements the gradient of the value function h(x)"""


class ControlAffineCBF(CBF):
    def __init__(self, dynamics, params, **kwargs) -> None:
        super().__init__(dynamics, params, **kwargs)

    def lie_derivatives(self, state, time):
        """
        :param state: state
        :param control: control
        :param time: time
        :return: f(x) g(x)
        """
        grad_vf = self._grad_vf(state, time)
        f = self.dynamics.open_loop_dynamics(state, time)
        g = self.dynamics.control_jacobian(state, time)
        Lf = np.atleast_1d(grad_vf @ f)
        Lg = np.atleast_2d(grad_vf @ g)
        return Lf, Lg
