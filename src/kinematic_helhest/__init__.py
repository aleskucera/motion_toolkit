"""Kinematic Helhest twin — a fast, differentiable, purely kinematic model of the
Helhest Junior skid-steer robot for planning.

Layout:
  engine/      The Warp/CUDA runtime — the main code path (import the API from here).
  reference/   Numpy finite-difference oracle — verification only, never the runtime.
  planning/    MPPI planner on the engine.
  viz/         glfw/OpenGL viewers + shared rendering toolkit.
  model, data, heightmap, friction   Shared geometry / scenes / fields.

The root package stays import-light on purpose: importing `kinematic_helhest`
does NOT pull in Warp, so the numpy reference is usable on its own. Reach into a
subpackage for what you need, e.g. `from kinematic_helhest.engine import step_kernel`.
"""
