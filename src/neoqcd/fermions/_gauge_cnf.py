"""The pure-gauge flow-field (CNF / Path Gradient) machinery used by the fermion package.

Same role as `_gauge.py` (the single point of contact with the pure-gauge modules), for the PG
training of `pg.py`: the fields, the pure-gauge loss, the ladder action, the portable checkpoint
and the distributed replica exchange are imported from here, never modified. Kept apart from
`_gauge.py` only to avoid an edit collision; the two may be merged.
"""

from neoqcd.cnf import (  # noqa: F401
    CloverChairLinkDrift,
    apply_velocity,
    coeff_to_matrix,
    lie_tangent,
    path_gradient_loss,
    path_gradient_residual,
)
from neoqcd.cnf_io import (  # noqa: F401
    DEFECT_PATCH_ARCHITECTURES,
    build_cnf,
    defect_patch_buffer,
    load_cnf_checkpoint,
    save_cnf_checkpoint,
)
from neoqcd.cnf_training import ladder_action  # noqa: F401
from neoqcd.distributedPT import ParallelTempering, PTConfig  # noqa: F401
