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
        return (
            self.vf(state, time) >= 0
        )  # TODO: This is safe with respect to CBF, not with respect to obstacle (more safety from obstacle)

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


class ImplicitCBF(abc.ABCMeta):
    def __init__(self, dynamics, params, **kwargs) -> None:
        self.dynamics = dynamics
        self.backup_policy = kwargs.get(
            "backup_policy", lambda x, t: np.zeros(self.dynamics.control_dims)
        )

    @abc.abstractmethod
    def backup_vf(self, state, time):
        """
        Implements the value function of the backup set h(x)
        """

    @abc.abstractmethod
    def _grad_backup_vf(self, state, time):
        """Implements the gradient of the value function of the backup set h(x)"""

    @abc.abstractmethod
    def grad_f_cl(self, state, time):
        """Implements the gradient of the closed loop dynamics under the
        backup policy pi_0(x) -> Handderived"""

    @abc.abstractmethod
    def safety_vf(self, state, time):
        """Implements the value function h(x) defining the safety set (can have multiple)"""

    @abc.abstractmethod
    def _grad_safety_vf(self, state, time):
        """Implements the gradient of the value function of the safety set h(x)"""

    def sensitivity_dt(self, Q, state, time):
        """ode presenting sensitivity matrix dQ_dt = Df_cl(\phi_t^u(x_0) * Q(t)"""
        return self.grad_f_cl(state, time) @ Q
