from cbf_opt.dynamics import Dynamics, ControlAffineDynamics
from cbf_opt.cbf import CBF, ControlAffineCBF
import numpy as np
import jax.numpy as jnp


def test_asif(asif_inst):
    from cbf_opt.asif import ASIF

    assert isinstance(asif_inst, ASIF)
    assert isinstance(asif_inst.dynamics, Dynamics)
    assert isinstance(asif_inst.cbf, CBF)
    assert asif_inst.dynamics == asif_inst.cbf.dynamics
    assert callable(asif_inst.alpha), "Alpha should be a function, e.g. lambda x: \gamma * x!"
    assert isinstance(asif_inst.alpha(np.random.rand()), float)
    x = np.random.rand(asif_inst.dynamics.n_dims)
    t = np.random.rand()
    if hasattr(asif_inst, "nominal_policy"):
        nom_pol = asif_inst.nominal_policy(x, t)
        assert isinstance(nom_pol, np.ndarray) or isinstance(nom_pol, jnp.DeviceArray)
        assert nom_pol.shape[-1] == asif_inst.dynamics.control_dims


def test_control_affine_asif(asif_inst):
    from cbf_opt.asif import ControlAffineASIF

    assert isinstance(asif_inst, ControlAffineASIF)
    test_asif(asif_inst)
    assert isinstance(asif_inst.dynamics, ControlAffineDynamics)
    assert isinstance(asif_inst.cbf, ControlAffineCBF)
    if asif_inst.umin is not None:
        assert asif_inst.umin.shape == (asif_inst.dynamics.control_dims,)
    if asif_inst.umax is not None:
        assert asif_inst.umax.shape == (asif_inst.dynamics.control_dims,)
    if asif_inst.umax is not None and asif_inst.umin is not None:
        assert (asif_inst.umax >= asif_inst.umin).all()
