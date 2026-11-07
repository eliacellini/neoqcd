"""The single point of contact with the pure-gauge modules.

The fermion package never imports neoqcd.* directly: it takes what it needs from here, so
that a later reorganisation of the gauge code changes this file only.
"""

from neoqcd.sun_utils import (  # noqa: F401
    SUN_dagger,
    SUN_haar,
    SUN_identity,
    SUN_mul,
    SUN_trace,
    SU3_unitarize,
    mean_plaquette,
    nstaple,
    plaquette_fields,
    pstaple,
)
from neoqcd.theory import Wilson_action  # noqa: F401
from neoqcd.smearing_jacobian import exp_coefficients  # noqa: F401
from neoqcd.theory import plaquette_weight  # noqa: F401
from neoqcd.mcmc import HBOR  # noqa: F401
from neoqcd.utils import create_mask  # noqa: F401
from neoqcd.smearing import CouplingLayer, ResidualCouplingLayer  # noqa: F401
from neoqcd.flow import FlowPars  # noqa: F401
from neoqcd.utils import Defect, translate_periodic_replica  # noqa: F401
from neoqcd.replica_history import replica_observable_series, rung_indices, summarize_history  # noqa: F401
