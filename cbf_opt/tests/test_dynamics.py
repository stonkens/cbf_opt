import numpy as np


# TODO: Generic test for dynamics, without having to need to have a specific dynamics class
# This should be linearized_ct_dynamics, linearizid_dt_dynamics, step

def test_dynamics(dyn_inst):
    state = np.random.rand(dyn_inst.n_dims)
    control = np.random.rand(dyn_inst.control_dims)
    time = np.random.rand(1)[0]
    assert dyn_inst(state, control, time).shape == (dyn_inst.n_dims,)
    assert dyn_inst.state_jacobian(state, control, time).shape == (dyn_inst.n_dims, dyn_inst.n_dims)
    assert dyn_inst.control_jacobian(state, control, time).shape = (dyn_inst.n_dims, dyn_inst.control_dims)


def test_control_affine_dynamics(dyn_inst):
    state = np.random.rand(dyn_inst.n_dims)
    control = np.random.rand(dyn_inst.control_dims)
    time = np.random.rand()
    assert dyn_inst.control_matrix(state, time).shape == (dyn_inst.n_dims, dyn_inst.control_dims)
    assert dyn_inst.open_loop_dynamics(state, time).shape == (dyn_inst.n_dims,)
    test_dynamics(dyn_inst)



    

