import numpy as np
import cbf_opt


def test_cbf(cbf_inst):
    from cbf_opt.cbf import CBF

    assert isinstance(cbf_inst, CBF)  # FIXME: Do we want to impose this restriction?
    assert isinstance(cbf_inst.dynamics, cbf_opt.dynamics.Dynamics)
    state = np.random.rand(cbf_inst.dynamics.n_dims)
    time = np.random.rand()
    assert isinstance(cbf_inst.vf(state, time), float)
    assert cbf_inst._grad_vf(state, time).shape == (cbf_inst.dynamics.n_dims,)

    dx = 1e-3 * np.random.rand(cbf_inst.dynamics.n_dims)
    grad_V = cbf_inst._grad_vf(state, time)
    V = cbf_inst.vf(state, time)
    dV_expected = grad_V @ dx
    dV_actual = cbf_inst.vf(state + dx, time) - V
    assert np.isclose(dV_expected, dV_actual, atol=1e-6)


def test_control_affine_cbf(cbf_inst):
    from cbf_opt.cbf import ControlAffineCBF

    assert isinstance(cbf_inst, ControlAffineCBF)  # FIXME: Do we want to impose this restriction?
    test_cbf(cbf_inst)
    assert isinstance(cbf_inst.dynamics, cbf_opt.dynamics.ControlAffineDynamics)
    state = np.random.rand(cbf_inst.dynamics.n_dims)
    time = np.random.rand()
    Lf, Lg = cbf_inst.lie_derivatives(state, time)
    assert Lf.ndim == 1
    assert Lg.ndim == 2
    assert Lf.shape == (1,)
    assert Lg.shape == (1, cbf_inst.dynamics.control_dims)


def test_backup_controller(ctrl_inst):
    assert isinstance(ctrl_inst.dynamics, cbf_opt.dynamics.Dynamics)
    state = np.random.rand(ctrl_inst.dynamics.n_dims)
    time = np.random.rand()
    assert ctrl_inst.umin.shape == (ctrl_inst.dynamics.control_dims,)
    assert ctrl_inst.umax.shape == (ctrl_inst.dynamics.control_dims,)
    assert ctrl_inst.policy(state, time).shape == (ctrl_inst.dynamics.control_dims,)
    assert ctrl_inst.grad_policy(state, time).shape == (
        ctrl_inst.dynamics.control_dims,
        ctrl_inst.dynamics.n_dims,
    )


def test_implicit_cbf(cbf_inst):
    assert isinstance(cbf_inst.dynamics, cbf_opt.dynamicsDynamics)
    test_backup_controller(cbf_inst.backup_controller)
    test_cbf(cbf_inst.safety_cbf)
    state = np.random.rand(cbf_inst.dynamics.n_dims)
    time = np.random.rand()
    assert isinstance(cbf_inst.backup_vf(state, time), float)
    assert isinstance(cbf_inst.safety_vf(state, time), float)
    assert cbf_inst._grad_backup_vf(state, time).shape == (cbf_inst.dynamics.n_dims,)
    assert cbf_inst._grad_safety_vf(state, time).shape == (cbf_inst.dynamics.n_dims,)
    assert isinstance(cbf_inst.vf(state, time), float)

    dx = 1e-3 * np.random.rand(cbf_inst.dynamics.n_dims)
    grad_V = cbf_inst._grad_backup_vf(state, time)
    V = cbf_inst.backup_vf(state, time)
    dV_expected = grad_V @ dx
    dV_actual = cbf_inst.backup_vf(state + dx, time) - V
    assert np.isclose(dV_expected, dV_actual, atol=1e-4)

    dx = 1e-3 * np.random.rand(cbf_inst.dynamics.n_dims)
    grad_V = cbf_inst._grad_safety_vf(state, time)
    V = cbf_inst.safety_vf(state, time)
    dV_expected = grad_V @ dx
    dV_actual = cbf_inst.safety_vf(state + dx, time) - V
    assert np.isclose(dV_expected, dV_actual, atol=1e-6)


def test_implicit_control_affine_cbf(cbf_inst):
    assert isinstance(cbf_inst.dynamics, cbf_opt.dynamics.ControlAffineDynamics)
    test_implicit_cbf(cbf_inst)
    test_control_affine_cbf(cbf_inst.safety_cbf)
    # state = np.random.rand(cbf_inst.dynamics.n_dims)
    # time = np.random.rand()
    # Lf, Lg = cbf_inst.lie_derivatives(state, time)  # TODO: Implement with sensitivity matrix
    # assert Lf.ndim == 1
    # assert Lg.ndim == 2
    # assert Lf.shape == (2,)
    # assert Lg.shape == (1, cbf_inst.dynamics.control_dims)
