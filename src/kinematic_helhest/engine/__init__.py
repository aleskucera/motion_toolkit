"""The Warp/CUDA runtime engine — the differentiable kinematic simulator.

This is the main code path (the numpy `reference/` package is the verification
oracle only). Public API:

    from kinematic_helhest.engine import (
        RobotParams, SolverParams, Simulator, init_state_kernel, step_kernel,
    )

`init_state`/`step`/`settle`/`clearances` are Warp kernels/funcs launched with
`wp.launch`; `Simulator` owns the device terrain/buffers and is the single entry
point for forward rollouts (feed it a device elevation array via `set_terrain`).
The implicit settle adjoint (`@wp.func_grad(settle)`) lives in `engine.step` and registers on
import, so gradients work automatically. The oracle/FD verification harness lives
in the top-level `tests/engine/` package (run e.g. `python -m tests.engine.step`).
"""

from .robot import Robot
from .robot import RobotParams
from .simulator import Simulator
from .step import clearances
from .step import init_state_kernel
from .step import settle
from .step import Solver
from .step import SolverParams
from .step import step_kernel
from .terrain import Grid
from .terrain import GridParams
from .terrain import sample_field
from .terrain import sample_height_grad
from .terrain import sample_normal
