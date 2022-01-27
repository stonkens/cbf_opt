import abc
import cvxpy as cp
import numpy as np


class ASIF(metaclass=abc.ABCMeta):
    def __init__(self, dynamics, cbf, **kwargs) -> None:
        self.dynamics = dynamics
        self.cbf = cbf
        self.alpha = kwargs.get("alpha", lambda x: x)
        self.verbose = kwargs.get("verbose", False)
        self.solver = kwargs.get("solver", "OSQP")
        self.nominal_policy = kwargs.get(
            "nominal_policy", lambda x, t: np.zeros(self.dynamics.control_dim)
        )

    @abc.abstractmethod
    def __call__(self, state, nominal_control, time):
        """Implements the active safety invariance filter"""


class ControlAffineASIF(ASIF):
    def __init__(self, dynamics, cbf, **kwargs) -> None:
        super().__init__(dynamics, cbf, **kwargs)
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

    def set_constraint(self, Lf_h, Lg_h, h):
        self.b.value = self.alpha(h) + Lf_h
        self.A.value = Lg_h

    def __call__(self, state, nominal_control=None, time=0.0):
        solver_failure = False
        if not hasattr(self, "QP"):
            self.setup_optimization_problem()
        h = self.cbf.vf(state, time)
        Lf_h, Lg_h = self.cbf.lie_derivatives(state, time)
        self.set_constraint(Lf_h, Lg_h, h)

        if nominal_control is not None:
            assert nominal_control.shape == (self.dynamics.control_dims,)
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


class ImplicitASIF(abc.ABCMeta):
    def __init__(self, dynamics, cbf, params, **kwargs) -> None:
        self.dynamics = dynamics
        self.cbf = cbf
        self.dt = params["dt"]
        self.T_backup = params["T_backup"]
        self.n_cbf_constraints = self.T_backup // self.dt
        self.alpha_backup = kwargs.get("alpha_backup", lambda x: x)
        self.alpha_safety = kwargs.get("alpha_safety", lambda x: x)
        self.nominal_policy = kwargs.get("nominal_policy", lambda x, t: 0)

    @abc.abstractmethod
    def __call__(self, state, nominal_control=None, time=0.0):
        """Implements the active safety invariance filter"""


class ImplicitControlAffineASIF(ImplicitASIF):
    # TODO: Not verified yet
    def __init__(self, dynamics, cbf, params, **kwargs) -> None:
        super().__init__(dynamics, cbf, params, **kwargs)
        self.filtered_control = cp.Variable(self.dynamics.control_dims)
        self.nominal_control = cp.Parameter(self.dynamics.control_dims)
        self.b = cp.Parameter(self.n_cbf_constraints + 2)
        self.A = cp.Parameter((self.n_cbf_constraints + 2, self.dynamics.control_dims))
        """
        min || u - u_des ||^2
        s.t. A @ u + b >= 0
        """
        self.obj = cp.Minimize(cp.norm(self.filtered_control - self.nominal_control, 2) ** 2)
        self.constraints = [self.A @ self.filtered_control + self.b >= 0]
        # TODO: Add umin and umax
        self.QP = cp.Problem(self.obj, self.constraints)

    def __call__(self, state, nominal_control=None, time=0):
        # grad_safety = self.cbf._grad_safety_vf(state, time)  # TODO: change to not have _grad_vf
        # grad_backup = self.cbf._grad_backup_vf(state, time)  # TODO: change to not have _grad_vf
        states, Qs, times = self.rollout_backup(self, state, time)
        A = np.zeros(self.A.shape)
        b = np.zeros(self.b.shape)
        grad_V = np.zeros((self.n_cbf_constraints + 2, self.dynamics.n_dims))
        for i in range(self.n_cbf_constraints + 1):
            grad_V[i] = self.cbf._grad_safety_vf(states[i], times[i]).T @ Qs[i]
            A[i] = grad_V[i] @ self.dynamics.control_jacobian(state, time)
            b[i] = grad_V[i] @ self.dynamics.open_loop_dynamics(state, time) + self.alpha_safety(
                self.cbf.safety_vf(states[i], times[i])
            )
        grad_V[-1] = self.cbf._grad_backup_vf(states[-1], times[-1]) @ Qs[-1]
        A[-1] = grad_V[-1] @ self.dynamics.control_jacobian(state, time)
        b[-1] = grad_V[-1] @ self.dynamics.open_loop_dynamics(state, time) + self.alpha_backup(
            self.cbf.backup_vf(states[-1], times[-1])
        )

        self.A.value = A
        self.b.value = b

        if nominal_control is not None:
            assert nominal_control.shape == (self.dynamics.control_dims,)
            self.nominal_control.value = nominal_control
        else:
            self.nominal_control.value = self.nominal_policy(state, time)

        self.QP.solve()
        # TODO: Relax solution if not solved with large penalty on constraints
        # TODO: Check if the solution is feasible
        return np.atleast_1d(self.filtered_control.value)

    def rollout_backup(self, state, time):
        states = [state]
        Q = np.eye(self.dynamics.state_dims)
        Qs = [Q]
        t = time
        times = [t]
        for i in range(self.n_cbf_constraints):
            dQ_dt = self.cbf.sensitivity_dt(Q, state, time)
            f = self.dynamics(state, self.cbf.backup_policy(state, time), time)
            state += self.dt * f
            Q += self.dt * dQ_dt
            Qs.append(Q)
            states.append(state)
            t += self.dt
            times.append(t)

        return states, Qs, times


class GeneralizedASIF(ASIF):
    def __init__(self, dynamics, cbf, **kwargs) -> None:
        super().__init__(dynamics, cbf, **kwargs)
        self.beta0 = kwargs.get("beta0", 0.0)
        self.penalty_coeff = kwargs.get("penalty_coeff", 1.0)


class GeneralizedControlAffineASIF(GeneralizedASIF, ControlAffineASIF):
    def __init__(self, dynamics, cbf, **kwargs) -> None:
        super().__init__(dynamics, cbf, **kwargs)
        self.beta = cp.Variable(1)
        self.obj += cp.Minimize(self.penalty_coeff * (self.beta - self.beta0) ** 2)
        self.constraints[0] = self.A @ self.filtered_control + self.b >= self.beta

    def get_beta(self):
        return self.beta.value
