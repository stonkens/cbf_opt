import abc
from typing import Any, Tuple, Dict
import numpy as np
from cbf_opt import Dynamics, ControlAffineDynamics
from cbf_opt.tests import test_cbf


class CBF(metaclass=abc.ABCMeta):
    def __init__(
        self, dynamics: Dynamics, params: Dict[str, Any], test: bool = True, **kwargs
    ) -> None:
        self.dynamics = dynamics
        if test:
            test_cbf.test_cbf(self)

    def is_safe(self, state: np.ndarray, time: float = 0.0) -> bool:
        """Evaluates h(x, t) >= 0"""
        return self.vf(state, time) >= 0

    def is_unsafe(self, state: np.ndarray, time: float = 0.0) -> bool:
        """Evaluates h(x, t) < 0"""
        return self.vf(state, time) < 0

    @abc.abstractmethod
    def vf(self, state: np.ndarray, time: float = 0.0) -> float:
        """Implements the value function h(x)"""

    @abc.abstractmethod
    def _grad_vf(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the gradient of the value function h(x)"""

    def vf_dt_partial(self, state: np.ndarray, time: float = 0.0) -> float:
        """Implements the partial derivative of the time derivative of the value function h(x)
        In general, we consider time-invariant CBFs, hence defaults to 0."""
        return 0.0

    def vf_dt(self, state: np.ndarray, control: np.ndarray, time: float = 0.0) -> float:
        """Implements the time derivative of the value function h(x)"""
        return self.vf_dt_partial(state, time) + self._grad_vf(state, time) @ self.dynamics(
            state, control, time
        )


class ControlAffineCBF(CBF):
    def __init__(
        self, dynamics: ControlAffineDynamics, params: Dict[str, Any], test: bool = True, **kwargs
    ) -> None:
        super().__init__(dynamics, params, test, **kwargs)
        if test:
            test_cbf.test_control_affine_cbf(self)

    def lie_derivatives(
        self, state: np.ndarray, time: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Lie derivatives of the dynamics at (state, time)
        :return: L_{f(x)} h, L_{g(x)} h with h the value function vf
        """
        grad_vf = self._grad_vf(state, time)
        f = self.dynamics.open_loop_dynamics(state, time)
        g = self.dynamics.control_matrix(state, time)
        try:
            Lf = np.atleast_1d(grad_vf @ f)
            Lg = np.atleast_2d(grad_vf @ g)
        except ValueError:
            Lf = np.einsum("ij,ij->i", grad_vf, f)
            Lg = np.einsum("ij, ijk->ik", grad_vf, g)
        return Lf, Lg


class ExponentialControlAffineCBF(ControlAffineCBF):
    def __init__(
        self, dynamics: ControlAffineDynamics, params: Dict[str, Any], test: bool = True, **kwargs
    ) -> None:
        """Only covers degree 2 Exponential CBF, i.e. Lg Lf \neq 0"""
        super().__init__(dynamics, params, test, **kwargs)
        self.Lf = kwargs.get("Lf")
        self.Lf2 = kwargs.get("Lf2")
        self.LgLf = kwargs.get("LgLf")
        self.alpha2 = kwargs.get("alpha2")

    def lie_derivatives(
        self, state: np.ndarray, time: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        Lf = np.atleast_1d(self.Lf(state, time) + self.alpha2 * self.Lf2(state, time))
        Lg = np.atleast_2d(self.LgLf(state, time))
        return Lf, Lg


class ImplicitCBF(CBF):
    def __init__(
        self, dynamics: Dynamics, params: Dict[str, Any], test: bool = True, **kwargs
    ) -> None:
        super().__init__(dynamics, params, test, **kwargs)
        self.backup_controller = kwargs.get("backup_controller")  # FIXME: Remove from here
        if test:
            test_cbf.test_implicit_cbf(self)

    def vf(self, x0: np.ndarray, t0: float = 0.0, break_unsafe: bool = False) -> float:
        # This is only used for the "hacky" version --> Is it?
        ts = np.arange(0, self.backup_controller.T_backup, self.dynamics.dt) + t0

        hs = np.zeros((ts.shape[0] + 2))
        val_curr = self.safety_vf(x0, ts[0])
        hs[0] = val_curr
        x = x0

        for i, t in enumerate(ts):
            if break_unsafe and val_curr < 0:
                return np.min(hs)

            action = self.backup_controller.policy(x, t)
            x = self.dynamics.step(x, action)
            val_curr = self.safety_vf(x, t)
            hs[i + 1] = val_curr

        hs[-1] = self.backup_vf(x, t0 + self.backup_controller.T_backup)
        return np.min(hs)

    def _grad_vf(self, state, time):
        raise NotImplementedError("Vf is not differentiable")

    @abc.abstractmethod
    def backup_vf(self, state: np.ndarray, time: float = 0.0) -> float:
        """
        Implements the value function of the backup set h(x)
        """

    # FIXME: How to define vf_dt_partial for Implicit CBFs? Is there any work on this?

    @abc.abstractmethod
    def _grad_backup_vf(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the gradient of the value function of the backup set h(x)"""

    @abc.abstractmethod
    def safety_vf(self, state: np.ndarray, time: float = 0.0) -> float:
        """Implements the value function h(x) defining the safety set (can have multiple)"""

    @abc.abstractmethod
    def _grad_safety_vf(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the gradient of the value function of the safety set h(x)"""


# TOTEST
class ControlAffineImplicitCBF(ImplicitCBF):
    def __init__(
        self, dynamics: ControlAffineDynamics, params: Dict[str, Any], test: bool = True, **kwargs
    ):
        super().__init__(dynamics, params, test, **kwargs)

    def lie_derivatives(
        self,
        state: np.ndarray,
        sensitivity: np.ndarray,
        time: float = 0.0,
        backup_set: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: L_{f(x)} V, L_{g(x)}
        """
        if backup_set:
            grad_vf = self._grad_backup_vf(state, time) @ sensitivity
        else:
            grad_vf = (
                self._grad_safety_vf(state, time) @ sensitivity
            )  # FIXME: Calculation seems slightly different for one of the two, to be checked!

        f = self.dynamics.open_loop_dynamics(state, time)
        g = self.dynamics.control_matrix(state, time)
        Lf = np.atleast_1d(grad_vf @ f)
        Lg = np.atleast_2d(grad_vf @ g)
        return Lf, Lg


# TOTEST
class BackupController:
    def __init__(self, dynamics: Dynamics, T_backup: float, test: bool = True, **kwargs):
        self.dynamics = dynamics
        self.T_backup = T_backup
        self.umin = kwargs.get("umin")
        self.umax = kwargs.get("umax")
        if test:
            test_cbf.test_backup_controller(self)

    @abc.abstractmethod
    def policy(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Implements the backup policy pi_0(x) -> Handderived"""

    @abc.abstractmethod
    def grad_policy(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Implements the gradient of the backup policy pi_0(x) -> Handderived"""

    def grad_f_cl(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Implements the gradient of the closed loop dynamics under the
        backup policy pi_0(x) -> Analytic expression"""
        action = self.policy(x, t)
        A = self.dynamics.state_jacobian(x, action, t)  # TODO: Might also require the control
        B = self.dynamics.control_matrix(x, action, t)
        return A + B @ self.grad_policy(x, t)

    def rollout_backup(self, x0: np.ndarray, t0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Verify shapes +1 or +2
        ts = np.arange(0, self.T_backup, self.dynamics.dt) + t0
        xs = np.zeros((ts.shape[0] + 1, self.dynamics.n_dims))

        xs[0] = x0
        for i, t in enumerate(ts):
            action = self.policy(xs[i], t)
            xs[i + 1] = self.dynamics.step(xs[i], action)

        return xs, ts

    def rollout_backup_w_sensitivity_matrix(
        self, x0: np.ndarray, t0: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: Verify shapes +1 or +2
        xs, ts = self.rollout_backup(x0, t0)
        Qs = np.zeros((ts.shape[0] + 1, self.dynamics.n_dims, self.dynamics.n_dims))
        Qs[0] = np.eye(self.dynamics.n_dims)
        for i, t in enumerate(ts):
            dQdt = self.sensitivity_dt(Qs[i], xs[i], t)
            Qs[i + 1] = Qs[i] + dQdt * self.dynamics.dt
        return xs, Qs, ts

    def sensitivity_dt(self, Q: np.ndarray, state: np.ndarray, time: float) -> np.ndarray:
        """ode presenting sensitivity matrix dQ_dt = Df_cl(\phi_t^u(x_0)) * Q(t)"""
        return self.grad_f_cl(state, time) @ Q
