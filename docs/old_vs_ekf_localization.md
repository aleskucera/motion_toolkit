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
          │    _pred         │    │  F      = jacobian_F_6d(ekf.x,  │
          │  → sweep_delta   │    │             u, omega_z=ωz)       │
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
              │    scale = (rms/rms_nom)² × (N_nom/N_inl)              │
              │    R_adaptive = R_ICP × scale     ← adaptive noise      │
              │                                                         │
              │    applied = ekf.update_icp(z, R=R_adaptive,           │
              │                             chi2_thresh=...)            │
              │     S = H P⁻ Hᵀ + R_adaptive                          │
              │     y = z − H ekf.x                                    │
              │     if yᵀ S⁻¹ y > chi2_thresh → skip (gate)           │
              │     else: K = P⁻ Hᵀ S⁻¹                               │
              │           ekf.x += K @ y   ← soft blend, NOT override  │
              │                                                         │
              │    if not applied:                                      │
              │      _consecutive_chi2_rejects += 1                    │
              │      if _consecutive_chi2_rejects >= max_rejects:      │
              │        force-accept (chi2_thresh=0) → snap to map      │
              │        _consecutive_chi2_rejects = 0                   │
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
**adaptive measurement noise** `R_adaptive = R_ICP × scale` (scaled by the ICP
fitness: RMS residual and inlier count) and a **Mahalanobis χ²(3) innovation gate**
that skips updates whose innovation is statistically implausible given the filter's
current uncertainty. A **force-accept escape hatch** fires after
`icp_chi2_max_rejects` consecutive gate rejections, snapping the filter back to the
ICP measurement to prevent the filter from silently diverging.

### Map-bias caveat

The "map is immune to EKF drift" guarantee holds on **accepted ICP frames** only. On
a localizer **reject** (`outcome.status == "rejected"`), `outcome.pose` falls back to
`world_T_base_pred` — the odom/gyro delta applied to the previous fused (`world_T_base`)
pose, which does include EKF influence. In that case `map_T_base = outcome.pose` is
effectively EKF-derived, and if the filter is diverging the map will reflect it. The
chi²-gate force-accept escape bounds this scenario by snapping the filter back to the
ICP measurement before the ICP seed drifts far enough to cause sustained localizer
rejects.

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
| **ICP measurement noise** | n/a | adaptive: `R_ICP × (rms/rms_nom)² × (N_nom/N_inl)` |
| **Innovation gate** | n/a | Mahalanobis χ²(3); skip if `yᵀS⁻¹y > icp_chi2_thresh` |
| **Gate escape hatch** | n/a | force-accept after `icp_chi2_max_rejects` consecutive skips |
| **Accumulator code** | identical | identical |

---

## Bag replay behaviour

The EKF predict step is driven by two inputs:

- **Translation** (`_prev_meas_wheel`): measured wheel velocities from `/joint_states`.
  The same signal in live operation and bag replay, so the prediction tracks the
  actual motion in both cases.
- **Yaw** (`_gyro_wz_mean`): the base-frame IMU gyro rate averaged over the
  inter-cloud window, replacing the wheel-differential yaw estimate. The gyro is
  immune to wheel slip and lateral-dynamics model error, which caused the EKF heading
  to lag ICP by up to 9° during turns — the χ² gate would then reject valid ICP
  corrections for the duration of the turn. Falls back to the wheel differential when
  no IMU samples are available.

The previous design used the MPPI planner's commanded output (`_prev_cmd_model`),
which diverged during bag replay because the live planner generated commands for a
different trajectory than the one in the bag.
