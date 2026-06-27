"""Numpy reference implementation (the finite-difference oracle).

This is the verification ground truth the Warp engine (engine/) is checked
against — NOT the runtime path. `placement`/`state`/`rollout`/`twist`/`turning`
are the numpy physics; `eval_phase*` are the phase verification harnesses;
`drive` is the numpy interactive viewer. Shared infrastructure (heightmap, model,
friction, data) stays at the package top level since both the reference and the
Warp engine use it.
"""
