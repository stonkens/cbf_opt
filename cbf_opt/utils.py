from hj_reachability import sets
import hj_reachability as hj
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from cbf_opt.dynamics import ControlAffineDynamics
from cbf_opt.cbf import CBF, ControlAffineCBF

# TOTEST
class HJControlAffineDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """Provides portability between cbf_opt and hj_reachability Dynamics definitions"""

    def __init__(self, dynamics, **kwargs):
        assert isinstance(dynamics, ControlAffineDynamics)

        self.dynamics = dynamics
        control_mode = kwargs.get("control_mode", "max")
        disturbance_mode = kwargs.get("disturbance_mode", "min")
        control_space = kwargs.get(
            "control_space",
            sets.Box(
                jnp.atleast_1d(-jnp.ones(self.dynamics.control_dims)),
                jnp.atleast_1d(jnp.ones(self.dynamics.control_dims)),
            ),
        )
        disturbance_space = kwargs.get(
            "disturbance_space",
            sets.Box(
                jnp.atleast_1d(-jnp.zeros(self.dynamics.disturbance_dims)),
                jnp.atleast_1d(jnp.zeros(self.dynamics.disturbance_dims)),
            ),
        )
        super().__init__(
            control_mode=control_mode,
            disturbance_mode=disturbance_mode,
            control_space=control_space,
            disturbance_space=disturbance_space,
        )

    def open_loop_dynamics(self, state, time=None):
        return jnp.array(self.dynamics.open_loop_dynamics(state, time))

    def control_jacobian(self, state, time=None):
        return jnp.array(self.dynamics.control_matrix(state, time))

    def disturbance_jacobian(self, state, time=None):
        return jnp.array(self.dynamics.disturbance_jacobian(state, time))


# TOTEST
class TabularCBF(CBF):
    def __init__(self, dynamics, grid: hj.grid.Grid, params: dict = dict(), **kwargs) -> None:
        super().__init__(dynamics, params, **kwargs)
        self.grid = grid
        self.grid_states_np = np.array(self.grid.states)
        self.grid_shape = self.grid.shape
        self.vf_table = None
        self.orig_cbf = None
        self.grad_vf_table = None

    def vf(self, state, time):
        assert self.vf_table is not None, "Requires instantiation of vf_table"
        return self.grid.interpolate(self.vf_table, state)

    def _grad_vf(self, state, time):
        if self.grad_vf_table is None:
            self.grad_vf_table = self.grid.grad_values(self.vf_table)

        return self.grid.interpolate(self.grad_vf_table, state)

    # TOTEST
    def tabularize_cbf(self, orig_cbf: CBF, time=0.0):
        """
        Tabularizes a control-affine CBF.
        """
        self.orig_cbf = orig_cbf
        assert isinstance(self.orig_cbf, CBF)
        assert self.orig_cbf.dynamics == self.dynamics

        self.vf_table = np.zeros(self.grid.shape)

        for i in tqdm(range(self.grid_shape[0])):
            if self.grid.ndim == 1:
                self.vf_table[i] = self.orig_cbf.vf(self.grid_states_np[i], time)
            else:
                for j in range(self.grid_shape[1]):
                    if self.grid.ndim == 2:
                        self.vf_table[i, j] = self.orig_cbf.vf(self.grid_states_np[i, j], time)
                    else:
                        for k in range(self.grid_shape[2]):
                            if self.grid.ndim == 3:
                                self.vf_table[i, j, k] = self.orig_cbf.vf(
                                    self.grid_states_np[i, j, k], time
                                )
                            else:
                                for l in range(self.grid_shape[3]):
                                    if self.grid.ndim == 4:
                                        self.vf_table[i, j, k, l] = self.orig_cbf.vf(
                                            self.grid_states_np[i, j, k, l], time
                                        )
                                    else:
                                        for m in range(self.grid_shape[4]):
                                            if self.grid.ndim == 5:
                                                self.vf_table[i, j, k, l, m] = self.orig_cbf.vf(
                                                    self.grid_states_np[i, j, k, l, m], time
                                                )
                                            else:
                                                for n in range(self.grid_shape[5]):
                                                    if self.grid.ndim == 6:
                                                        self.vf_table[
                                                            i, j, k, l, m, n
                                                        ] = self.orig_cbf.vf(
                                                            self.grid_states_np[i, j, k, l, m, n],
                                                            time,
                                                        )
                                                    else:
                                                        raise NotImplementedError(
                                                            "Only up to 6 dimensions supported"
                                                        )
        self.vf_table = self.vf_table.T
