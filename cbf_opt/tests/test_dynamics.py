import numpy as np
from cbf_opt.dynamics import Dynamics, ControlAffineDynamics

# TODO: Generic test for dynamics, without having to need to have a specific dynamics class
# This should be linearized_ct_dynamics, linearizid_dt_dynamics, step


def test_dynamics(dyn_inst):
    assert isinstance(dyn_inst, Dynamics)
    state = np.random.rand(dyn_inst.n_dims)
    control = np.random.rand(dyn_inst.control_dims)
    time = np.random.rand(1)[0]
    assert dyn_inst(state, control, time).shape == (dyn_inst.n_dims,)  # For any system
    assert dyn_inst.step(state, control, time, scheme="fe").shape == (
        dyn_inst.n_dims,
    )  # For any system
    assert dyn_inst.step(state, control, time, scheme="rk4").shape == (
        dyn_inst.n_dims,
    )  # For any system
    assert dyn_inst.control_jacobian(state, control, time).shape == (
        dyn_inst.n_dims,
        dyn_inst.control_dims,
    )
    try:
        assert dyn_inst.state_jacobian(state, control, time).shape == (
            dyn_inst.n_dims,
            dyn_inst.n_dims,
        )
        A_d, B_d = dyn_inst.linearized_dt_dynamics(state, control, time)
        assert A_d.shape == (dyn_inst.n_dims, dyn_inst.n_dims)
        assert B_d.shape == (dyn_inst.n_dims, dyn_inst.control_dims)
        state2 = state + 1e-2 * np.random.rand(dyn_inst.n_dims)
        control2 = control + 1e-2 * np.random.rand(dyn_inst.control_dims)
        state_next = dyn_inst.step(state, control, time)
        state2_next = dyn_inst.step(state2, control, time)
        assert np.isclose(
            state2_next - state_next, A_d @ (state2 - state) + B_d @ (control2 - control), atol=1e-6
        ).all(), "Taylor expansion not accurate, linearized dynamics might be wrong"

    except NotImplementedError:
        pass  # Not mandatory to implement


def test_control_affine_dynamics(dyn_inst):
    test_dynamics(dyn_inst)
    assert isinstance(dyn_inst, ControlAffineDynamics)
    state = np.random.rand(dyn_inst.n_dims)
    time = np.random.rand()
    assert dyn_inst.open_loop_dynamics(state, time).shape[-1] == (dyn_inst.n_dims,)
    assert dyn_inst.control_matrix(state, time).shape[-2:] == (
        dyn_inst.n_dims,
        dyn_inst.control_dims,
    )
