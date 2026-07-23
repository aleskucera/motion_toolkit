# Localization loop: `elevation_node` vs `elevation_node_ekf`

Both nodes run the same per-frame pipeline (scan → ICP → accumulate → maps → plan).
The key difference is **how they derive the pose(s)** used to place scans in the map
and broadcast the robot's location. `elevation_node` has one pose — the raw ICP result.
`elevation_node_ekf` produces two: **`map_T_base`** (raw ICP, used for map writing and
carving) and **`world_T_base`** (EKF-blended, used for TF, planning, and the next ICP
seed). This separation is what makes the EKF filter's smoothing safe: it never contaminates
the accumulated point cloud.

---

## `elevation_node` — odom + ICP, direct

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Per-frame inputs: odom_msg, cloud_msg, imu_buffer                          │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  localizer.predict()  │
          │                       │
          │  translation: odom Δ  │
          │  rotation:    gyro Δ  │   (gyro rotation delta integrated from the
          │                       │    buffered IMU; replaces wheel-odom yaw
          └──────────┬────────────┘    which is unreliable under skid)
                     │
                     │  world_T_base_pred  (odom-predicted SE(3) pose)
                     │  sweep_delta        (used to deskew the scan)
                     ▼
          ┌───────────────────────┐
          │  Scan pre-processing  │
          │  z-crop, self-filter, │
          │  range-crop, deskew,  │
          │  outlier removal      │
          └──────────┬────────────┘
                     │  scan_wp  (denoised base-frame scan, on GPU)
                     ▼
          ┌───────────────────────────────────────┐
          │  localizer.update()  — ICP             │
          │                                       │
          │  seed:      world_T_base_pred          │
          │  target:    accumulated map (map_wp)   │
          │  prior:     gravity_up (IMU tilt)      │
          │                                       │
          │  → outcome.pose   (accepted SE(3))     │
          │    or world_T_base_pred (on reject)    │
          └──────────┬────────────────────────────┘
                     │
                     │  world_T_base = outcome.pose
                     │           ▲
                     │           └─ raw ICP result; no further modification
                     ▼
          ┌───────────────────────┐
          │  world_scan =         │
          │  transform_points(    │
          │    scan_wp,           │
          │    world_T_base)      │   ← THE only pose that enters the map
          └──────────┬────────────┘
                     │
                     ▼
          ┌───────────────────────┐
          │  acc.step()           │
          │  (carve + merge +     │
          │   voxelise + crop)    │   accumulated_map = self.map_wp
          └───────────────────────┘
```

### Key property
`world_T_base` **is** the ICP result. There is no second estimator; the scan is
placed into the map exactly where ICP put it.

---

## `elevation_node_ekf` — odom + ICP + EKF physics filter

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Per-frame inputs: odom_msg, cloud_msg, imu_buffer, _prev_meas_wheel         │
└─────────────────────────────────────────────┬────────────────────────────────┘
                                              │
               ┌──────────────────────────────┴─────────────────────────────┐
               │  (A) MOTION PREDICTION — two independent paths             │
               └──────────────────────────────┬─────────────────────────────┘
                                              │
          ┌──────────────────┐    ┌───────────┴──────────────────────┐
          │ localizer.       │    │  EKF predict step                │
          │ predict()        │    │                                  │
          │                  │    │  u  = _prev_meas_wheel           │
          │  odom Δ trans    │    │       (measured /joint_states)   │
          │  gyro Δ rot      │    │  ωz = _gyro_wz_mean(t0, t1)     │ ← IMU gyro (slip-immune)
          │                  │    │       fallback: wheel diff       │
          │                  │    │  x_pred = predict_q6d(ekf.x,    │
          │  → world_T_base  │    │             u, omega_z=ωz)       │
          │    _pred         │    │  F = jacobian_F_6d_analytical(  │
          │  → sweep_delta   │    │             ekf.x, x_pred, DT)   │
          └────────┬─────────┘    │  r = clamp(dt/DT, 0.5, 3.0)     │  ← dt from cloud stamps
                   │              │  scale Δxy,Δψ,F[0:2,2] by r     │
                   │              │  ekf.predict(F,x_pred,q_scale=r) │
                   │              │   → ekf.x updated (x,y,ψ,ẋ,ẏ,ψ̇)│
                   │              │   → ekf.P updated  (Q×r growth)  │
                   │              └──────────────────────────────────┘
                   │  world_T_base_pred  (EKF-fused prev pose ⊕ odom Δ)
                   │  sweep_delta        (odom Δ only — used to deskew)
                   ▼
          ┌───────────────────────┐
          │  Scan pre-processing  │
          │  (identical to plain) │
          └──────────┬────────────┘
                     │  scan_wp
                     ▼
          ┌───────────────────────────────────────┐
          │  localizer.update()  — ICP             │
          │                                       │
          │  seed:    world_T_base_pred            │  ← EKF-fused prev ⊕ odom Δ
          │  target:  map_wp                       │
          │  prior:   gravity_up                   │
          │                                       │
          │  → outcome.pose  (or fallback)         │
          └──────────┬────────────────────────────┘
                     │
              ┌──────┴──────────────────────────────────────────────────┐
              │  (B) EKF MEASUREMENT UPDATE                             │
              │                                                         │
              │  if outcome.status == "ok":                             │
              │    z   = [x_icp, y_icp, ψ_icp]   ← absolute ICP pose  │
              │    rms = outcome.rms_residual_m                         │
              │    scale = max((rms/rms_nom)² × (N_nom/N_inl), 0.25)  │
              │    R_adaptive = R_ICP × scale     ← adaptive noise      │
              │                                                         │
              │    ekf.update_icp(z, R=R_adaptive)                     │
              │     S = H P⁻ Hᵀ + R_adaptive                          │
              │     y = z − H ekf.x                                    │
              │     K = P⁻ Hᵀ S⁻¹                                     │
              │     ekf.x += K @ y   ← soft blend, NOT override        │
              └──────┬──────────────────────────────────────────────────┘
                     │
                     │  world_T_base = _splice_planar(
                     │      outcome.pose,          ← z, roll, pitch from ICP
                     │      ekf.x[0],              ← x     from EKF (blended)
                     │      ekf.x[1],              ← y     from EKF (blended)
                     │      ekf.x[2])              ← yaw   from EKF (blended)
                     │
                     │  map_T_base = outcome.pose   ← raw ICP (no EKF blend)
                     │
                     │  localizer.set_corrected_pose(world_T_base)
                     │      ↑ feeds EKF-fused pose back so the NEXT frame's
                     │        ICP seed uses it instead of the raw ICP result
                     │
              ┌──────┴──────────────────┐    ┌────────────────────────────┐
              │  MAP WRITING            │    │  EXPORT (TF / planning)    │
              │  (map_T_base)           │    │  (world_T_base)            │
              └──────┬──────────────────┘    └────────────────────────────┘
                     ▼
          ┌───────────────────────┐
          │  world_scan =         │
          │  transform_points(    │
          │    scan_wp,           │
          │    map_T_base)        │   ← raw ICP pose; on accepted frames EKF bias
          │                       │     does not enter the map. On localizer rejects
          │                       │     map_T_base falls back to the EKF-derived seed
          │                       │     (see "Map-bias caveat" below).
          └──────────┬────────────┘
                     │
                     ▼
          ┌───────────────────────┐
          │  acc.step()           │
          │  (identical to plain) │   accumulated_map = self.map_wp
          └───────────────────────┘
```

### Key property
`elevation_node_ekf` produces **two poses per frame**:

- **`map_T_base = outcome.pose`** — the raw ICP result (or odom fallback on reject).
  This is what places the scan in the accumulated cloud and what the carving
  rays are cast from. On accepted ICP frames it is identical to what the plain
  node uses, so the map is immune to EKF drift on those frames.

- **`world_T_base = _splice_planar(outcome.pose, ekf.x[0:3])`** — the Kalman-blended
  pose: z/roll/pitch from ICP (gravity-anchored terrain tilt), x/y/yaw from the EKF
  state (physics prediction + ICP update, K < I). This is exported to TF, planning,
  and feeds the next ICP seed via `localizer.set_corrected_pose()`.

The design is the **raw-map / filtered-export separation**: the map is built from
raw sensor data only, so the ICP measurement cannot be biased by the filter's own
history. The EKF measurement update itself is a standard **absolute-pose** loosely-
coupled fusion (`z = [x_icp, y_icp, ψ_icp]`, `H = [I₃|0₃]`). The update uses an
**adaptive measurement noise** `R_adaptive = R_ICP × scale` where
`scale = max((rms/rms_nom)² × (N_nom/N_inl), 0.25)` — scaled by ICP fitness
(RMS residual and inlier count) so noisy alignments get less weight and clean ones
get more, with a floor at 0.25 preventing R from collapsing to zero on exceptionally
clean scans.

### Map-bias caveat

The "map is immune to EKF drift" guarantee holds on **accepted ICP frames** only. On
a localizer **reject** (`outcome.status == "rejected"`), `outcome.pose` falls back to
`world_T_base_pred` — the odom/gyro delta applied to the previous fused (`world_T_base`)
pose, which does include EKF influence. In that case `map_T_base = outcome.pose` is
effectively EKF-derived, and if the filter is diverging the map will reflect it. The
sustained-reject reset machinery (`reset_after_rejects`) bounds this scenario by
wiping and re-seeding the map before the ICP seed drifts far enough to cause permanent
localizer rejects.

---

## Side-by-side summary

| | `elevation_node` | `elevation_node_ekf` |
|---|---|---|
| **ICP seed** | odom Δ applied to previous ICP pose | odom Δ applied to previous **EKF-fused** pose |
| **Map writing pose** | `outcome.pose` (raw ICP) | `outcome.pose` (raw ICP — identical on accepted frames) |
| **Post-ICP export pose** | `outcome.pose` directly | `_splice_planar(outcome.pose, ekf.x)` |
| **x/y/yaw source (export)** | raw ICP | EKF-blended (physics predict + ICP update) |
| **z/roll/pitch source** | raw ICP | raw ICP (same) |
| **Fallback when ICP rejects** | odom-predicted pose | EKF-predicted pose (physics model) |
| **Physics model input** | n/a | `_prev_meas_wheel` (wheel speeds) + `_gyro_wz_mean` (IMU ωz, slip-immune yaw) |
| **Localizer seed override** | n/a | `set_corrected_pose(world_T_base)` each frame |
| **ICP measurement noise** | n/a | adaptive: `R_ICP × scale`, `scale = max((rms/rms_nom)² × (N_nom/N_inl), 0.25)` |
| **Accumulator code** | identical | identical |

---

## Bag replay behaviour

The EKF predict step is driven by two inputs:

- **Translation** (`_prev_meas_wheel`): measured wheel velocities from `/joint_states`.
  The same signal in live operation and bag replay (`ekf-demo` play_topics must
  include `/joint_states`; without it predict gets `u = [0,0,0]`), so the prediction
  tracks the actual motion in both cases.
- **Yaw** (`_gyro_wz_mean`): the base-frame IMU gyro rate averaged over the
  inter-cloud window, replacing the wheel-differential yaw estimate. The gyro is
  immune to wheel slip and lateral-dynamics model error, which caused the EKF heading
  to lag ICP by up to 9° during turns — the χ² gate would then reject valid ICP
  corrections for the duration of the turn. Falls back to the wheel differential when
  no IMU samples are available.

The previous design used the MPPI planner's commanded output (`_prev_cmd_model`),
which diverged during bag replay because the live planner generated commands for a
different trajectory than the one in the bag.

---

## EKF noise matrices and tunable parameters

All values live in `elevation_node_ekf.py` — the matrices near the top of the file,
the ROS parameters in `_declare_parameters()`.

### State vector

```
x = [x,  y,  ψ,  ẋᵂ,  ẏᵂ,  ψ̇]   (6-DOF)
     pos  pos  yaw  world-frame velocities
```

### Initial state covariance P₀  (diagonal, `[6×6]`)

```python
_SIG_P0 = [0.10 m,  0.10 m,  2.0°,  0.30 m/s,  0.30 m/s,  0.20 rad/s]
```

| State | 1-σ | Rationale |
|---|---|---|
| x, y | 0.10 m | first ICP fix typically within 10 cm of the odom seed |
| ψ | 2.0° | gyro heading is accurate at boot; small initial heading uncertainty |
| ẋᵂ, ẏᵂ | 0.30 m/s | velocities not directly measured; generous to let the predict step dominate early |
| ψ̇ | 0.20 rad/s | same rationale |

### Process noise Q  (diagonal, `[6×6]`)

```python
_SIG_Q = [0.02 m,  0.02 m,  0.5°,  0.15 m/s,  0.15 m/s,  0.10 rad/s]
```

Represents uncertainty added per predict step (one LiDAR frame ≈ 0.1 s).
Velocity rows are generous because F[:,3:6] = 0 — the simulator re-derives
velocities from the wheel-speed input `u` at each step rather than integrating them.

| State | 1-σ/step | Rationale |
|---|---|---|
| x, y | 0.02 m | ~2 cm/step model error for straight-line sliding terrain |
| ψ | 0.5° | small heading model error; gyro covers most of it |
| ẋᵂ, ẏᵂ | 0.15 m/s | unobservable states; kept ≈ P₀ so P_vv stays near its initial value |
| ψ̇ | 0.10 rad/s | same |

### ICP measurement noise R_ICP  (diagonal, `[3×3]`)

```python
_SIG_R_ICP = [0.05 m,  0.05 m,  1.0°]
```

Nominal (scale = 1) uncertainty of an ICP pose measurement. Used as the base
matrix that `R_adaptive = R_ICP × scale` scales.

| Component | 1-σ | Rationale |
|---|---|---|
| x, y | 0.05 m | typical lateral + longitudinal ICP position spread under normal conditions |
| ψ | 1.0° | heading from ICP is usually better than position; 1° is conservative |

### Adaptive measurement noise parameters (ROS params)

| Parameter | Default | Units | Description |
|---|---|---|---|
| `icp_r_rms_nom` | `0.018` | m | Nominal ICP RMS residual. Calibrated as geometric mean of observed median RMS across 3 bags (0.019 m, 0.015 m, 0.021 m) — minimax-optimal at 0.018 m (max cross-bag scale deviation 0.30). Set to the typical median RMS for your sensor and scene. |
| `icp_r_inl_nom` | `4800` | # points | Nominal inlier count. Calibrated to observed mean inlier count. |

`scale = max((rms / rms_nom)² × (N_nom / N_inl),  0.25)`

- `scale = 1.0` at nominal operating conditions → `R_adaptive = R_ICP`
- `scale > 1` for a noisy/sparse scan → larger R → smaller K → less ICP weight
- `scale < 1` for a clean/dense scan → smaller R → larger K → more ICP weight
- Floor at **0.25** prevents R from collapsing below ¼ R_ICP even on perfect scans

Effective Kalman gain K_xy observed across two bags: **0.36–0.42** (takes roughly
one-third of the ICP position innovation, discards two-thirds as predicted by the
physics model). This was verified by bag replay in July 2026.

---

## Debug topics (`publish_ekf_debug`)

`elevation_node_ekf` publishes three informational topics when the ROS parameter
`publish_ekf_debug` is `true` (default).  Nothing in the stack subscribes to them;
they exist purely for monitoring and tuning.

| Topic | Type | Rate | Contents |
|---|---|---|---|
| `ekf/odom` | `nav_msgs/Odometry` | every frame | Fused pose (EKF x/y/yaw, ICP z/roll/pitch), world-frame velocity rotated into base_frame, and the full 6×6 covariance blocks (z/roll/pitch diagonal = 1e6 sentinel — not filtered). |
| `ekf/nis_icp` | `std_msgs/Float32` | accepted ICP frames only | Normalised Innovation Squared (NIS) of the ICP measurement update. |
| `ekf/diagnostics` | `diagnostic_msgs/DiagnosticArray` | every frame | Per-frame scalar summary: `status`, `nis`, `innov_x_m`, `innov_y_m`, `innov_yaw_deg`, `r_scale`, `rms_residual_m`, `num_inliers`, `dt_ratio`, `consecutive_rejects`. Level `WARN` on rejected/sparse frames or when NIS exceeds the χ²(3) 99th percentile. |

### Reading NIS

The ICP update observes three DOF (`[x, y, ψ]`), so NIS = yᵀ S⁻¹ y is **χ²(3)** distributed
when the filter is consistent (Q and R match the true noise):

| NIS value | Interpretation |
|---|---|
| mean ≈ 3 | consistent — Q/R are well tuned |
| sustained > 7.81 (95th percentile) | innovation is too large → R_ICP too small, or Q too small (filter over-confident), or predict is biased |
| sustained < 1 | innovation is too small → R_ICP too large (filter ignores ICP) |
| > 11.34 (99th percentile) | single-frame: flags in `ekf/diagnostics` as WARN |

Watch `ekf/nis_icp` in PlotJuggler or `rqt_plot`; a rolling mean well above 7 during
normal driving is a signal to raise `_SIG_R_ICP` or lower `_SIG_Q` (or vice-versa).
`ekf/odom` covariance gives a sanity check that P does not collapse or diverge over time.
