import abc
from typing import Any, Tuple, Dict
import numpy as np
from cbf_opt import Dynamics, ControlAffineDynamics
from cbf_opt.tests import test_cbf
import itertools


class CBF(metaclass=abc.ABCMeta):
    def __init__(self, dynamics: Dynamics, params: Dict[str, Any], test: bool = True, **kwargs) -> None:
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
        return self.vf_dt_partial(state, time) + self._grad_vf(state, time) @ self.dynamics(state, control, time)


class ControlAffineCBF(CBF):
    def __init__(self, dynamics: ControlAffineDynamics, params: Dict[str, Any], test: bool = True, **kwargs) -> None:
        super().__init__(dynamics, params, test, **kwargs)
        if test:
            test_cbf.test_control_affine_cbf(self)

    def lie_derivatives(self, state: np.ndarray, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Lie derivatives of the dynamics at (state, time)
        :return: L_{f(x)} h, L_{g(x)} h with h the value function vf
        """
        grad_vf = self._grad_vf(state, time)

        # self.dynamics.all_params is False if it is empty (aka no parameters for system)
        keys, values = zip(*self.dynamics.all_params.items()) if self.dynamics.all_params else ((), ())
        extremum_values = (v if isinstance(v, list) else [v] for v in values)
        extremums = [dict(zip(keys, v)) for v in itertools.product(*extremum_values)]

        Lfs, Lgs, Lws = [], [], []

        for extremum in extremums:
            self.dynamics.params = extremum
            f = self.dynamics.open_loop_dynamics(state, time)
            g = self.dynamics.control_matrix(state, time)
            w = self.dynamics.disturbance_matrix(state, time)

            try:
                Lf = np.atleast_1d(grad_vf @ f)
                Lg = np.atleast_2d(grad_vf @ g)
                Lw = np.atleast_2d(grad_vf @ w)
            except (ValueError, TypeError):
                Lf = np.einsum("ij,ij->i", grad_vf, f)
                Lg = np.einsum("ij, ijk->ik", grad_vf, g)
                Lw = np.einsum("ij, ijk->ik", grad_vf, w)
            Lfs.append(Lf)
            Lgs.append(Lg)
            Lws.append(Lw)

        return np.array(Lfs).swapaxes(0, 1), np.array(Lgs).swapaxes(0, 1), np.array(Lws).swapaxes(0, 1)


class ExponentialControlAffineCBF(ControlAffineCBF):
    def __init__(self, dynamics: ControlAffineDynamics, params: Dict[str, Any], test: bool = True, **kwargs) -> None:
        """Only covers degree 2 Exponential CBF, i.e. Lg Lf \neq 0"""
        super().__init__(dynamics, params, test, **kwargs)
        self.Lf = kwargs.get("Lf")
        self.Lf2 = kwargs.get("Lf2")
        self.LgLf = kwargs.get("LgLf")
        self.alpha2 = kwargs.get("alpha2")

    def lie_derivatives(self, state: np.ndarray, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        Lf = np.atleast_1d(self.Lf(state, time) + self.alpha2 * self.Lf2(state, time))
        Lg = np.atleast_2d(self.LgLf(state, time))
        return Lf, Lg


# TOTEST
class BackupController:
    def __init__(self, dynamics: Dynamics, T_backup: float, test: bool = True, **kwargs):
        self.dynamics = dynamics
        self.T_backup = T_backup
        self.umin = kwargs.get("umin", -np.infty * np.ones(dynamics.control_dims))
        self.umax = kwargs.get("umax", np.infty * np.ones(dynamics.control_dims))
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
        A = self.dynamics.state_jacobian(x, action, t)
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
        """ode presenting sensitivity matrix
        dQ_dt = Df_cl(\phi_t^u(x_0)) * Q(t)"""
        return self.grad_f_cl(state, time) @ Q


class ImplicitCBF(CBF):
    def __init__(
        self,
        dynamics: Dynamics,
        params: Dict[str, Any],
        backup_controller: BackupController,
        safety_cbf: CBF,
        test: bool = True,
        **kwargs
    ) -> None:
        self.backup_controller = backup_controller
        self.safety_cbf = safety_cbf
        super().__init__(dynamics, params, test=False, **kwargs)  # FIXME: test=False to avoid notimplemented error
        if test:
            test_cbf.test_implicit_cbf(self)

    def vf(self, x0: np.ndarray, t0: float = 0.0, break_unsafe: bool = False) -> float:
        # This is only used for the "hacky" version --> Is it?
        # FIXME: Make amenable to batched x0
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

    @abc.abstractmethod
    def _grad_backup_vf(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the gradient of the value function of the backup set h(x)"""

    def safety_vf(self, state: np.ndarray, time: float = 0.0) -> float:
        """Implements the value function h(x) defining the safety set"""
        return self.safety_cbf.vf(state, time)

    def _grad_safety_vf(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Implements the gradient of the value function of the safety set h(x)"""
        return self.safety_cbf._grad_vf(state, time)


# TOTEST
class ControlAffineImplicitCBF(ImplicitCBF):
    """
    Implements the control affine implicit backup function
    Opting to not have it as child of ControlAffineCBF for simplicity
    """

    def __init__(
        self,
        dynamics: ControlAffineDynamics,
        params: Dict[str, Any],
        backup_controller: BackupController,
        safety_cbf: ControlAffineCBF,
        test: bool = True,
        **kwargs
    ):
        super().__init__(dynamics, params, backup_controller, safety_cbf, test, **kwargs)
        if test:
            test_cbf.test_implicit_control_affine_cbf(self)

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

        # ts = np.arange(0, self.backup_controller.T_backup, self.dynamics.dt) + t0

        # hs = np.zeros((*x0[..., 0].shape, ts.shape[0] + 2))
        # val_curr = self.safety_vf(x0, ts[0])
        # hs[..., 0] = val_curr
        # x = x0
        # vf = np.zeros_like(hs[..., 0])
        # cond = np.zeros_like(hs[..., 0], dtype=bool)

        # for i, t in enumerate(ts):
        #     lax.cond()
        #     if break_unsafe and val_curr < 0:

        #         return np.min(hs)
        #     def next_hs(x, t):
        #         action = self.backup_controller.policy(x, t)
        #         x = self.dynamics.step(x,)
        #     action = self.backup_controller.policy(x, t)
        #     x = self.dynamics.step(x, action)
        #     val_curr = self.safety_vf(x, t)
        #     hs[i + 1] = val_curr

        # hs[-1] = self.backup_vf(x, t0 + self.backup_controller.T_backup)
        # return np.min(hs)
