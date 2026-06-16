"""Numpy reference implementation (the finite-difference oracle).

This is the verification ground truth the Warp engine (warp_engine/) is checked
against — NOT the runtime path. `state`/`rollout`/`twist`/`turning` are the numpy
physics; `eval_phase*` are the phase verification harnesses. Shared infrastructure
(heightmap, model, friction, data, placement, drive-rendering) stays at the package
top level since both the reference and the Warp engine use it.
"""
