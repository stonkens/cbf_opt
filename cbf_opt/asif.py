import abc
import cvxpy as cp
import numpy as np
from cbf_opt import dynamics as dynamics_f
from cbf_opt import cbf as cbf_f


class ASIF(metaclass=abc.ABCMeta):
    def __init__(self, dynamics: dynamics_f.Dynamics, cbf: cbf_f.CBF, **kwargs) -> None:
        self.dynamics = dynamics
        self.cbf = cbf
        self.alpha = kwargs.get("alpha", lambda x: x)
        self.verbose = kwargs.get("verbose", False)
        self.solver = kwargs.get("solver", "OSQP")
        self.nominal_policy = kwargs.get(
            "nominal_policy", lambda x, t: np.zeros(self.dynamics.control_dims)
        )
        assert isinstance(self.dynamics, dynamics_f.Dynamics)
        assert isinstance(self.cbf, cbf_f.CBF)

    @abc.abstractmethod
    def __call__(
        self, state: np.ndarray, nominal_control: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """Implements the active safety invariance filter"""


class ControlAffineASIF(ASIF):
    def __init__(
        self, dynamics: dynamics_f.ControlAffineDynamics, cbf: cbf_f.ControlAffineCBF, **kwargs
    ) -> None:
        super().__init__(dynamics, cbf, **kwargs)

        assert isinstance(self.dynamics, dynamics_f.ControlAffineDynamics)
        assert isinstance(self.cbf, cbf_f.ControlAffineCBF)

        self.filtered_control = cp.Variable(self.dynamics.control_dims)
        self.nominal_control = cp.Parameter(self.dynamics.control_dims)
        self.umin = kwargs.get("umin")
        self.umax = kwargs.get("umax")
        self.b = cp.Parameter(1)
        self.A = cp.Parameter((1, self.dynamics.control_dims))
        """
        min || u - u_des ||^2
        s.t. A @ u + b >= 0
        """
        self.obj = cp.Minimize(cp.norm(self.filtered_control - self.nominal_control, 2) ** 2)
        self.constraints = [self.A @ self.filtered_control + self.b >= 0]
        if self.umin:
            self.constraints.append(self.filtered_control >= self.umin)
        if self.umax:
            self.constraints.append(self.filtered_control <= self.umax)

    def setup_optimization_problem(self):
        self.QP = cp.Problem(self.obj, self.constraints)

    def set_constraint(self, Lf_h: np.ndarray, Lg_h: np.ndarray, h: float):
        self.b.value = self.alpha(h) + Lf_h
        self.A.value = Lg_h

    def __call__(self, state: np.ndarray, nominal_control=None, time: float = 0.0) -> np.ndarray:
        solver_failure = False
        if not hasattr(self, "QP"):
            self.setup_optimization_problem()
        h = self.cbf.vf(state, time)
        Lf_h, Lg_h = self.cbf.lie_derivatives(state, time)
        self.set_constraint(Lf_h, Lg_h, h)

        if nominal_control is not None:
            assert isinstance(nominal_control, np.ndarray) and nominal_control.shape == (
                self.dynamics.control_dims,
            )
            self.nominal_control.value = nominal_control
        else:
            self.nominal_control.value = self.nominal_policy(state, time)
        try:
            self.QP.solve(solver=self.solver, verbose=self.verbose)
        except cp.SolverError:
            solver_failure = True
        if self.QP.status in ["infeasible", "unbounded"] or solver_failure:
            # TODO: Add logging / printing
            if (self.umin is None) and (self.umax is None):
                return np.atleast_1d(self.nominal_control.value)
            else:
                if self.umin and self.umax:
                    # TODO: This should depend on "controlMode"
                    return np.atleast_1d(
                        np.int64(Lg_h >= 0) @ np.atleast_1d(self.umax)
                        + np.int64(Lg_h < 0) @ np.atleast_1d(self.umin)
                    )
                elif (Lg_h >= 0).all() and self.umax:
                    return np.atleast_1d(self.umax)
                elif (Lg_h <= 0).all() and self.umin:
                    return np.atleast_1d(self.umin)
                else:
                    return np.atleast_1d(self.nominal_control.value)

        # TODO: Relax solution if not solved with large penalty on constraints
        return np.atleast_1d(self.filtered_control.value)


class ImplicitASIF(metaclass=abc.ABCMeta):
    def __init__(
        self,
        dynamics: dynamics_f.Dynamics,
        cbf: cbf_f.ImplicitCBF,
        backup_controller: cbf_f.BackupController,
        **kwargs
    ) -> None:
        self.dynamics = dynamics
        self.cbf = cbf
        self.backup_controller = backup_controller
        assert isinstance(self.backup_controller, cbf_f.BackupController)
        self.verify_every_x = kwargs.get("verify_every_x", 1)
        self.n_backup_steps = int(self.backup_controller.T_backup / self.dynamics.dt)
        self.alpha_backup = kwargs.get("alpha_backup", lambda x: x)
        self.alpha_safety = kwargs.get("alpha_safety", lambda x: x)
        self.nominal_policy = kwargs.get("nominal_policy", lambda x, t: 0)

        # assert isinstance(self.dynamics, dynamics.Dynamics)
        # assert isinstance(self.cbf, cbf.ImplicitCBF)

    @abc.abstractmethod
    def __call__(self, state: np.ndarray, nominal_control=None, time: float = 0.0) -> np.ndarray:
        """Implements the active safety invariance filter"""


# TO TEST
class ImplicitControlAffineASIF(ImplicitASIF):
    def __init__(
        self,
        dynamics: dynamics_f.ControlAffineDynamics,
        cbf: cbf_f.ControlAffineImplicitCBF,
        backup_controller: cbf_f.BackupController,
        **kwargs
    ) -> None:
        super().__init__(dynamics, cbf, backup_controller, **kwargs)

        assert isinstance(self.dynamics, dynamics_f.ControlAffineDynamics)
        assert isinstance(self.cbf, cbf_f.ControlAffineImplicitCBF)
        self.filtered_control = cp.Variable(self.dynamics.control_dims)
        self.nominal_control = cp.Parameter(self.dynamics.control_dims)
        self.b = cp.Parameter(int(self.n_backup_steps / self.verify_every_x) + 2)
        self.A = cp.Parameter(
            (int(self.n_backup_steps / self.verify_every_x) + 2, self.dynamics.control_dims)
        )
        """
        min || u - u_des ||^2
        s.t. A @ u + b >= 0
        """
        self.obj = cp.Minimize(cp.norm(self.filtered_control - self.nominal_control, 2) ** 2)
        self.constraints = [self.A @ self.filtered_control + self.b >= 0]
        # TODO: Add umin and umax
        self.QP = cp.Problem(self.obj, self.constraints)

    def __call__(self, state: np.ndarray, nominal_control=None, time: float = 0.0) -> np.ndarray:
        # grad_safety = self.cbf._grad_safety_vf(state, time)  # TODO: change to not have _grad_vf
        # grad_backup = self.cbf._grad_backup_vf(state, time)  # TODO: change to not have _grad_vf
        states, Qs, times = self.backup_controller.rollout_backup_w_sensitivity_matrix(state, time)
        A = np.zeros(self.A.shape)
        b = np.zeros(self.b.shape)

        for i in range(self.n_backup_steps // self.verify_every_x):
            idx = i * self.verify_every_x
            b[i], A[i] = self.cbf.lie_derivatives(
                states[idx], Qs[idx], times[idx]
            )  # TODO: Fix why it shows error -> Do I use wrong terminology for super classes?
            b[i] += self.alpha_safety(self.cbf.safety_vf(states[idx], times[idx]))

        b[-2], A[-2] = self.cbf.lie_derivatives(states[-1], Qs[-1], times[-1])
        b[-2] += self.alpha_backup(self.cbf.safety_vf(states[-1], times[-1]))
        b[-1], A[-1] = self.cbf.lie_derivatives(
            states[-1], Qs[-1], times[-1], backup_set=True
        )  # TODO: Discuss with Yuxiao if there is an additional term for the nonbackup set -> YES TO IMPLEMENT
        b[-1] += self.alpha_backup(self.cbf.backup_vf(states[-1], times[-1]))

        self.A.value = A
        self.b.value = b

        if nominal_control is not None:
            assert isinstance(nominal_control, np.ndarray) and nominal_control.shape == (
                self.dynamics.control_dims,
            )
            self.nominal_control.value = nominal_control
        else:
            self.nominal_control.value = self.nominal_policy(state, time)

        self.QP.solve()
        # TODO: Relax solution if not solved with large penalty on constraints
        # TODO: Check if the solution is feasible
        return np.atleast_1d(self.filtered_control.value)


# TOTEST
class TradeoffFilter(ImplicitASIF):
    def __init__(
        self,
        dynamics: dynamics_f.Dynamics,
        cbf: cbf_f.ImplicitCBF,
        backup_controller: cbf_f.BackupController,
        **kwargs
    ):
        super().__init__(dynamics, cbf, backup_controller, **kwargs)
        self.beta = kwargs.get("beta", 10)
        self.decay_func = kwargs.get(
            "decay_func", lambda x, t, h: 1 - np.exp(-self.beta * np.maximum(h, 0))
        )

    def __call__(self, state: np.ndarray, nominal_control=None, time: float = 0.0) -> np.ndarray:
        assert self.decay_func is not None, "Decay function must be specified"
        h_curr = self.cbf.vf(state, time)
        # print(h_curr)
        filter_rate = self.decay_func(state, time, h_curr)

        if nominal_control is not None:
            assert isinstance(nominal_control, np.ndarray) and nominal_control.shape == (
                self.dynamics.control_dims,
            )
        else:
            nominal_control = self.nominal_policy(state, time)

        return np.atleast_1d(
            filter_rate * nominal_control
            + (1 - filter_rate) * self.backup_controller.policy(state, time)
        )


# TOTEST
# FIXME: Not for this code base
class GeneralizedASIF(ASIF):
    def __init__(self, dynamics: dynamics_f.Dynamics, cbf: cbf_f.CBF, **kwargs) -> None:
        super().__init__(dynamics, cbf, **kwargs)
        self.beta0 = kwargs.get("beta0", 0.0)
        self.penalty_coeff = kwargs.get("penalty_coeff", 1.0)


# TOTEST
# FIXME: Not for this code base
class GeneralizedControlAffineASIF(GeneralizedASIF, ControlAffineASIF):
    def __init__(
        self, dynamics: dynamics_f.ControlAffineDynamics, cbf: cbf_f.ControlAffineCBF, **kwargs
    ) -> None:
        super().__init__(dynamics, cbf, **kwargs)
        self.beta = cp.Variable(1)
        self.obj += cp.Minimize(self.penalty_coeff * (self.beta - self.beta0) ** 2)
        self.constraints[0] = self.A @ self.filtered_control + self.b >= self.beta

    def get_beta(self) -> float:
        return self.beta.value
