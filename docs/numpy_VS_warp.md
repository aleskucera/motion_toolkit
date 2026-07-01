# NumPy implementation (`src/kinematic_helhest/reference/`)

Single-threaded, pure NumPy. Entry points: `state.make_state` (init) and `state.step` (advance).

Each step follows **predict → project**:

1. **Build envelope** (once): morphologically dilate the raw heightmap by a spherical cap of radius `R` → wheel placement surface `surf`.
2. **Turning params**: sample friction `mu_i` at the three contact points, weight by normal loads `N_i` → friction-weighted ICR offset `x_icr` and turn-resistance factor `alpha`.
3. **Twist (predict)**: skid-steer formula from wheel speeds `omega = [wL, wR, w_rear]` → body-frame `(vx, vy, wz)`. `vy = −x_icr · wz`; forward speed divided by `alpha`.
4. **World velocity + Euler**: rotate `(vx, vy, 0)` by the current orientation matrix `R` (so climbing on a slope reduces horizontal progress), integrate `(x, y, yaw)` by `dt`.
5. **Settle (project)**: finite-difference Newton solve for `(z, pitch, roll)` at the new planar pose — drives wheel hub clearances to zero against `surf`. Warm-started from previous solution.
6. **Normal loads**: quasi-static 3×3 linear solve (vertical force balance + torque about CoM) → `N_i` [N] per wheel.
7. **Chassis clearance**: sample raw terrain under the body grid → minimum belly gap; negative → `valid = False` (high-centred).

# Warp implementation (`src/kinematic_helhest/engine/`)

GPU-parallel, batched. Runs thousands of rollouts simultaneously (MPPI). Same physics as the reference, split into three Warp functions per timestep.

Per step:

1. **`step_predict`**: build orientation `R = euler_zyx(yaw, pitch, roll)`; compute `normal_loads` (quasi-static solve); sample friction grid at contacts → friction-weighted `x_icr` and `alpha`; apply skid-steer twist; rotate to world frame; Euler-integrate → predicted `(xn, yn, yawn)`.
2. **`settle`**: analytic-Jacobian Newton on `(z, pitch, roll)` at the predicted planar pose (analytic `dR/dθ` and bilinear terrain gradients instead of finite differences). Carries an IFT adjoint for gradient backprop through the converged root.
3. **`step_finalize`**: write new `controlled` / `derived` state vectors; recompute `normal_loads`, chassis clearance against raw elevation, and settle residual for diagnostics.

Two rollout modes:
- **`ForwardSimulator`**: fused `rollout_kernel` — full trajectory state kept in registers, no autodiff, fastest path.
- **`DifferentiableSimulator`**: per-step kernel recorded on a `wp.Tape`; gradients flow through the IFT settle adjoint back to terrain elevation and friction grids.
