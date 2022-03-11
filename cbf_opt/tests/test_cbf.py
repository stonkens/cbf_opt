import numpy as np


def test_cbf(cbf_inst):
    state = np.random.rand(cbf_inst.dyn.n_dims)
    time = np.random.rand(1)[0]
    assert isinstance(cbf_inst.vf(state, time), float)
    assert cbf_inst._grad_vf.shape == (cbf_inst.dyn.n_dims,)
    # assert cbf_inst.is_safe(state, time)
    # assert cbf_inst.vf(state, time) >= 0
    # assert cbf_inst.vf_dt(state, time) >= 0
    # assert cbf_inst.vf_dt_partial(state, time) >= 0
    # assert cbf_inst.vf_dt(state, time) >= 0
    # assert cbf_inst.vf_dt_partial(state, time) >= 0
    # assert cbf_inst.vf_dt(state, time) >= 0
    # assert cbf_inst.vf_dt_partial(state, time) >= 0
    # assert cbf_inst.vf_dt(state, time) >= 0
    # assert cbf_inst.vf_dt_partial(state, time) >= 0
    # assert cbf_inst.vf_dt(state, time) >= 0
    # assert cbf_inst.vf_dt_partial(state, time) >= 0
    # assert cbf_inst.vf_dt(state, time) >= 0


def test_control_affine_cbf(cbf_inst):
    state = np.random.rand(cbf_inst.dyn.n_dims)
    time = np.random.rand()
    Lf, Lg = cbf_inst.lie_derivatives()
    assert Lf.ndim == 1
    assert Lg.ndim == 2
    assert Lf.shape == (cbf_inst.dyn.n_dims,)
    assert Lg.shape == (cbf_inst.dyn.n_dims, cbf_inst.dyn.control_dims)
    test_cbf(cbf_inst)


def test_implicit_cbf(cbf_inst):
    state = np.random.rand(cbf_inst.dyn.n_dims)
    time = np.random.rand(1)[0]
    assert isinstance(cbf_inst.backup_vf(state, time), float)
    assert isinstance(cbf_inst.safety_vf(state, time), float)
    assert cbf_inst._grad_backup_vf(state, time).shape == (cbf_inst.dyn.n_dims,)
    assert cbf_inst._grad_safety_vf(state, time).shape == (cbf_inst.dyn.n_dims,)

    assert isinstance(cbf_inst.vf(state, time), float)


def test_backup_controller(ctrl_inst):
    state = np.random.rand(ctrl_inst.dyn.n_dims)
    control = np.random.rand(ctrl_inst.dyn.control_dims)
    time = np.random.rand()
    assert ctrl_inst.umin.shape == (ctrl_inst.dyn.control_dims,)
    assert ctrl_inst.umax.shape == (ctrl_inst.dyn.control_dims,)
    assert ctrl_inst.policy(state, time).shape == (ctrl_inst.dyn.control_dims,)
    assert ctrl_inst.grad_policy(state, time).shape == (
        ctrl_inst.dyn.control_dims,
        ctrl_inst.dyn.n_dims,
    )
