# HOTFIX: turn-differential compensation (`plan_turn_boost`)

> **Status: HOTFIX / stopgap.** `plan_turn_boost` (in `control/command.py`, exposed as the
> `elevation_node` param `plan_turn_boost`) is a band-aid over a **drivetrain / motor-control
> defect**, not a real fix. It compensates for a symptom. Read this before you change it, remove
> it, or trust the robot's turning outdoors.

## Symptom

Outdoors, the robot **under-turns badly** — it rotates at only ~10–15 % of the yaw rate the MPPI
controller intends. It drives forward fine but "won't turn how MPPI wants." (First noticed on
`bags/outdoor_experiment{2,3,4}`.)

## What we RULED OUT (so nobody re-investigates these)

Measured against ICP-truth localization (verified against the gyro), all four of these are **fine**:

- **MPPI / the planner** — it commands the correct turn; direction is right (cmd-vs-actual corr
  +0.5…+0.7). (An earlier "robot turns the opposite way" claim was an *analysis* indexing bug, not
  real — differencing a wheel against the Unix timestamp. Retracted.)
- **The vehicle model / K_TURN** — using the *measured* wheel speeds, the effective turn gain
  α ≈ 2.2–2.5 vs the model's ~1.8, consistent across differential magnitudes and forward speeds.
  No deadband, no speed dependence.
- **The sign convention** — `/cmd_joints` is all-positive-forward; the LLC's internal mapping is
  correct. Forward and turn go the intended directions.
- **Localization** — ICP `/tf` yaw tracks the gyro at corr ~0.9, scale ~1.0.

## ROOT CAUSE (the real defect)

**The drivetrain realizes the commanded *forward* speed 1:1 but only ~49 % of the commanded *turn*
differential.** The two drive motors equalize a commanded wheel-speed difference under load.

Evidence (binned means ± SE, pooled over the 3 outdoor bags; `/joint_setpoint` command vs
`/joint_states` measured, same motor-side units):

| command component | measured / commanded |
|---|---|
| forward  `(R + L)` | **1.00** (lands exactly on the proportional line) |
| turn     `(R − L)` | **0.49** (systematically half; error bars far too small for noise) |

So when MPPI commands e.g. `L=4, R=6` (a turn), the motors deliver ~`L=4.5, R=5.5`: the *average*
(forward) is right, the *spread* (turn) is halved → little yaw. Likely mechanisms (unconfirmed):
the two wheel velocity loops sharing a current/torque budget or coupled tuning, and/or the **driven
rear wheel** (commanded straight ahead) physically resisting the yaw.

## THE HOTFIX — what `plan_turn_boost` does

`condition_command` splits each command into forward (mean) + turn (differential) and multiplies
**only the differential** by `turn_boost`, leaving the forward speed untouched:

```
mean = (wl + wr) / 2
diff = (wr - wl) * turn_boost
cmd  = [mean - diff/2,  mean,  mean + diff/2]        # then slew-limit + clamp to plan_max_omega
```

Commanding a bigger spread means that after the drivetrain eats ~half of it, the *realized* spread
matches what MPPI intended, so the robot turns as planned.

- **Param:** `plan_turn_boost` (default **1.0 = off**), live-tunable (`ros2 param set /elevation
  plan_turn_boost 2.0`).
- **Starting value:** ~**2.0** (= 1 / 0.49) recovers the differential loss. If it still under-turns,
  go to ~2.5–2.8 to also cover the residual model understeer (measured α ~2.5 vs model 1.8). If it
  over-turns / oscillates (S-ing around the path), lower it.
- Terrain-dependent (the loss was measured on grass) — treat like `terrain` / `k_turn`.

## Why this is a HOTFIX, not a fix

- It compensates the **symptom** (halved differential), not the cause (motors won't hold a
  differential). The underlying defect is still there.
- It **stresses the motors harder** — commanding a larger differential (the inner wheel can go
  slightly negative on tight turns) to fight a drivetrain that resists it.
- It's an **environment-tuned magic number**, not physics. It will drift as terrain/load changes.

## WHAT TO ACTUALLY DO (when there is time)

1. **Fix the wheel velocity control loops** so they hold a commanded differential under load —
   investigate the shared current/torque budget and the velocity-loop tuning at the LLC/motor level.
2. **Isolate the rear wheel's contribution:** record a calibration drive with the **rear wheel
   disabled** and re-run the turn-tracking measurement (below). If the realized differential jumps
   toward 1:1, the rear wheel driving straight was fighting the yaw — decide whether to drive it
   differently during turns or leave it passive.
3. Once the drivetrain holds a differential, **set `plan_turn_boost` back to 1.0** and delete this
   hotfix path.
4. Longer term, replace all the manual per-terrain knobs (`k_turn`, `terrain`, `plan_turn_boost`)
   with **online estimation** of the turn model + differential realization from live (command →
   measured-wheel → yaw) data.

## How to re-measure / verify

The measurement that produced the numbers above (no node/GPU needed — gyro + wheels only):

- Record: `/joint_setpoint`, `/joint_states`, `/ouster/imu`, `/imu/data`, `/odom_2d` (see
  `ros/calibrate_turn.sh` for a lean recorder).
- Realization factor: bin `/joint_setpoint` `(R+L)` and `(R−L)` and plot the mean `/joint_states`
  response per bin — forward should be slope ~1, turn is the number to watch (was ~0.49).
- After enabling the boost, record a drive and re-run this: the realized differential should move
  toward 1:1, and the actual yaw should track the MPPI-intended yaw.
- Related tooling: `ros/calibrate_turn.sh` (turn-gain fit), the `wheel_sign_convention_calibration`
  memory.
