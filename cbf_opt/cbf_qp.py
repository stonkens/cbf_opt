import gym
import numpy as np

"""
Decide on how to set up the dynamics:
- Gym-like: __init__, step (evolve env one step forward), reset, sample_random_state, etc. (check openai gym)
- hj_reachability inspired: open_loop_dynamics, control_jacobian, disturbance_jacobian, etc., __call__ implements the forward dynamics f(x) + G_u(x) @ u + G_d(x) @ d --> for non-control affine figure out whether we just want the "system" to fail
"""
import cvxpy as cp


class SimulateDynamics:
    def __init__(self) -> None:
        pass
        # set default nominal policy to random and default asif as identity (both lambda functions)

    def step(self, state):
        action = self.asif(self.nominal_policy(state))
        return self.dynamics(state, action)

    def reset(self):
        # TODO: Sample an initial state
        return x0


class DynamicsModel:
    def __init__(self):
        pass

    def control_affine_dynamics(self, x, u):
        """
        :param x: state
        :param u: control
        :return: f(x) g(x)
        """
        pass


class CarFollowing(DynamicsModel):
    def __init__(self):
        super().__init__()
        self.n_dims = 3
        self.control_dims = 1
        self.mass = 100
        self.f0 = 5
        self.f1 = 10
        self.f2 = 5
        self.rolling_resistance = lambda x: self.f0 + self.f1 * x[1] + self.f2 * x[1] ** 2
        self.v0 = 20

    def __call__(self, x, u):
        f, g = self.control_affine_dynamics(x)
        return f + g @ u

    def control_affine_dynamics(self, x):
        """
        :param x: state
        :param u: control
        :return: f(x) g(x)
        """

        f = np.array([x[1], -1 / self.mass * self.rolling_resistance(x), self.v0 - x[1]])
        g = np.array([0, 1 / self.mass, 0])
        return f, g


class CBFFunctionality:
    def __init__(self, dynamics, **kwargs) -> None:
        self.dynamics = dynamics
        self.alpha = kwargs.get("alpha", lambda x: x)

    def _gradV(self, X):
        raise NotImplementedError()

    def value_function(self, x):
        raise NotImplementedError()

    def lie_derivatives(self, x):
        """
        :param x: state
        :return: f(x)
        """
        gradV = self._gradV(x)
        f, g = self.dynamics.control_affine_dynamics(x)
        Lf = np.array(
            [
                gradV @ f,
            ]
        )
        import pdb

        pdb.set_trace()
        Lg = gradV @ g
        return Lf, Lg

        # TODO: Do we want to batch it (possiblity to have multiple "scenarios")?


class SafeDistanceCBF(CBFFunctionality):
    def __init__(self, dynamics, **kwargs) -> None:
        super().__init__(dynamics, **kwargs)

    def value_function(self, x):
        """
        :param x: state
        :return: V(x)
        """
        return (
            x[2]
            - self.Th * x[1]
            - 0.5 * (x[1] - self.dynamics.v0) ** 2 / (self.dynamics.cd * self.dynamics.g)
        )

    def _gradV(self, x):
        return np.array(
            [
                0,
                -self.Th - (x[1] - self.dynamics.v0) / (self.dynamics.cd * self.dynamics.g),
                1,
            ]
        )


class ASIF:
    def __init__(self) -> None:
        self.u_filtered = cp.Variable(self.dynamics.n_controls)
        self.u_des = cp.Parameter(self.dynamics.n_controls)
        self.b = cp.Parameter(self.dynamics.n_dims)
        self.A = cp.Parameter(self.dynamics.n_dims, self.dynamics.n_controls)

        """
        min || u - u_des ||^2
        s.t. A @ u + b >= 0
        """
        obj = cp.Minimize(cp.norm(self.u_filtered - self.u_des, 2) ** 2)
        constraints = [self.A @ self.u_filtered + self.b >= 0]
        self.ASIF_QP = cp.Problem(obj, constraints)

    def __call__(self, x, u_nominal=None):
        """
        :param x: state
        :param u_nominal: nominal control
        :return: u_filtered
        """
        V = self.CBF.value_function(x)
        Lf_V, Lg_V = self.CBF.lie_derivatives(x)
        self.b.value = self.CBF.alpha(V) + Lf_V
        self.A.value = Lg_V

        if u_nominal is not None:
            assert u_nominal.shape == (self.dynamics.n_controls,)
            self.u_des.value = u_nominal
        else:
            self.u_des.value = self.CBF.nominal_policy(x)
        self.ASIF_QP.solve()
        # TODO: Check if the solution is feasible
        return self.u_filtered.value


def test_lie_derivatives():
    N_tests = 10
    # TODO: Sample N_tests for x, u from state space
    Lf_V, Lg_V = CBF.lie_derivatives(x)
    Vdot = Lf_V + Lg_V @ u
    delta_t = 1e-4
    V_now = CBF.V(x)
    xdot = dynamics(x, u)
    x_next = x + xdot * delta_t
    V_next = CBF.V(x_next)
    Vdot_fe = (V_next - V_now) / delta_t

    assert np.isclose(Vdot, Vdot_fe).all()


# TODO: Implement actually using the dynamics model and forward propogating (using forward euler)
def test_open_loop_dynamics():
    pass
