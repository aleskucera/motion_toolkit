# Helhest State Model for Filtering / EKF

## 1. State model

$$\mathbf{x}_t = \begin{pmatrix} x \\ y \\ \psi \end{pmatrix}, \qquad (z,\;\theta,\;\phi) \text{ terrain-derived}$$

### Turning parameters

$$w_i = \mu_i N_i, \qquad x_\text{ICR} = \frac{\sum_i w_i x_i}{\sum_i w_i}, \qquad \alpha = 1 + k \frac{\sum_i w_i}{mg}$$

### Body-frame twist from wheel speeds

Given the command $\boldsymbol{\omega} = (\omega_L,\, \omega_R,\, \omega_\text{rear})$:

$$\begin{aligned}
\dot{x}^B &= \frac{r(\omega_L + \omega_R)}{2} \\
\omega_z   &= \frac{r(\omega_R - \omega_L)}{2b\alpha} \\
\dot{y}^B &= -x_\text{ICR}\,\omega_z
\end{aligned}$$

The rear wheel is kinematically redundant and does not enter the twist.

### World-frame integration (forward Euler)

$$\begin{aligned}
\begin{pmatrix}\dot{x}^W \\ \dot{y}^W \\ 0\end{pmatrix} &= \mathbf{R}(\psi,\, \theta,\, \phi) \begin{pmatrix}\dot{x}^B \\ \dot{y}^B \\ 0\end{pmatrix} \\[6pt]
\mathbf{x}_{t+1} &= f(\mathbf{x}_t,\, \boldsymbol{\omega}_t) = \mathbf{x}_t + \begin{pmatrix}\dot{x}^W \\ \dot{y}^W \\ \omega_z\end{pmatrix}\Delta t
\end{aligned}$$

### Rotation matrix

$\mathbf{R} = R_z(\psi)\,R_y(\theta)\,R_x(\phi)$ written out (Z-Y-X intrinsic; nose-up pitch is negative; maps body-frame ${}^B$ to world-frame ${}^W$):

$$\mathbf{R}(\psi,\theta,\phi) = \begin{pmatrix} c_\psi c_\theta & c_\psi s_\theta s_\phi - s_\psi c_\phi & c_\psi s_\theta c_\phi + s_\psi s_\phi \\ s_\psi c_\theta & s_\psi s_\theta s_\phi + c_\psi c_\phi & s_\psi s_\theta c_\phi - c_\psi s_\phi \\ -s_\theta & c_\theta s_\phi & c_\theta c_\phi \end{pmatrix}$$

where $c_\alpha = \cos\alpha$, $s_\alpha = \sin\alpha$.

This nonlinear map $f$ is implemented exactly in `engine/step.py` (`rollout_kernel`) and is the process model for the filter.

---

## 2. Constants

Helhest Junior defaults (source: `src/helhest/model.py` and `src/helhest/dynamics.py`):

| Symbol | Value | Unit | Meaning |
|--------|-------|------|---------|
| $r$ | 0.35 | m | Wheel radius |
| $b$ | 0.365 | m | Half-track: lateral distance from body centreline to each front wheel hub |
| $l$ | 0.75 | m | Rear offset: longitudinal distance from body origin to rear wheel hub |
| $m$ | 106.2 | kg | Total vehicle mass |
| $g$ | 9.81 | m/s² | Gravitational acceleration |
| $k$ | 2.0 | — | Turn-resistance gain; empirically tuned — higher $k$ means more grip widens the effective track more aggressively |
| $\Delta t$ | 0.1 | s | Control timestep |
| $x_i$ | $[0,\; 0,\; -l]$ | m | Longitudinal body-frame coordinate of wheel $i$ (left, right, rear) |
| $\mathbf{R}(\psi,\theta,\phi)$ | — | — | Z-Y-X intrinsic rotation matrix: $R_z(\psi)\,R_y(\theta)\,R_x(\phi)$; maps body ${}^B$ to world ${}^W$; nose-up pitch is negative |
| $\mu_i$ | terrain-sampled | — | Friction coefficient at wheel $i$'s contact point (from the friction grid) |
| $N_i$ | quasi-static solve | N | Normal contact load at wheel $i$, from static equilibrium under gravity |
| $w_i$ | — | N | Grip weight of wheel $i$: $w_i = \mu_i N_i$ |
| $x_\text{ICR}$ | — | m | Longitudinal offset of the instantaneous centre of rotation; grip-weighted centroid of the wheel positions |
| $\alpha$ | $\ge 1$ | — | Effective track-widening factor; reduces the yaw rate produced by a given speed difference when total grip is high |

---

## 3. Jacobian for EKF linearisation

**[WARNING: non-exist in the codebase, recommended by AI]**

The process function $f(\mathbf{x}_t, \boldsymbol{\omega}_t)$ is nonlinear due to the rotation $\mathbf{R}(\psi,\theta,\phi)$ and the terrain-dependent turning parameters $(\alpha, x_\text{ICR})$. The EKF predict step requires the state-transition Jacobian:

$$F_t = \left.\frac{\partial f}{\partial \mathbf{x}}\right|_{\mathbf{x}_t,\,\boldsymbol{\omega}_t}$$

**On flat ground** ($\theta = \phi = 0$, uniform $\mu$), $\alpha$ and $x_\text{ICR}$ are constant and the Jacobian reduces to a closed-form 3×3 matrix in $\psi$, $\dot{x}^B$, $\dot{y}^B$, and $\omega_z$.

**On terrain**, $F_t$ is computed numerically via central differences — three additional one-step `ForwardSimulator` rollouts, one per state dimension with perturbation $\delta$:

$$F_t[:,\, j] \approx \frac{f(\mathbf{x}_t + \delta\,\mathbf{e}_j,\, \boldsymbol{\omega}_t) - f(\mathbf{x}_t - \delta\,\mathbf{e}_j,\, \boldsymbol{\omega}_t)}{2\delta}$$
