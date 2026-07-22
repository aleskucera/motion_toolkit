# Helhest state space model

## 1. State and input

$$\mathbf{q} = \begin{pmatrix} q_1 \\ q_2 \\ q_3 \\ q_4 \\ q_5 \\ q_6 \end{pmatrix} = \begin{pmatrix} x \\ y \\ \psi \\ \dot{x}^W \\ \dot{y}^W \\ \dot{\psi} \end{pmatrix}, \qquad \mathbf{u} = \begin{pmatrix} \omega_L \\ \omega_R \\ \omega_\text{rear} \end{pmatrix}$$

$q_4, q_5, q_6$ are world-frame velocities. This is a **kinematic** model: the velocities are algebraically determined by the wheel-speed input $\mathbf{u}$ and the current heading $q_3 = \psi$ — they carry no independent inertial dynamics.

---

## 2. Process model $\dot{\mathbf{q}} = f(\mathbf{q},\, \mathbf{u})$

### Position kinematics

$$\begin{aligned}
\dot{q}_1 &= q_4 \\
\dot{q}_2 &= q_5 \\
\dot{q}_3 &= q_6
\end{aligned}$$

### Velocity kinematics

The body-frame twist follows from the kinematic model (see [`motion_model.md`](motion_model.md)):

$$\begin{aligned}
v_x^B   &= \frac{r(\omega_L + \omega_R)}{2} \\[4pt]
q_6     &= \dot{\psi} = \frac{r(\omega_R - \omega_L)}{2b\alpha} \\[4pt]
v_y^B   &= -x_\text{ICR}\, q_6
\end{aligned}$$

Rotating to the world frame via $\mathbf{R}(\psi,\theta,\phi) = R_z(\psi)\,R_y(\theta)\,R_x(\phi)$ (Z-Y-X intrinsic; $\theta, \phi$ terrain-derived):

$$\begin{pmatrix} q_4 \\ q_5 \\ 0 \end{pmatrix} = \mathbf{R}(\psi,\,\theta,\,\phi) \begin{pmatrix} v_x^B \\ v_y^B \\ 0 \end{pmatrix}$$

**On flat ground** ($\theta = \phi = 0$) the rotation collapses to $R_z(\psi)$ and the expressions simplify to:

$$\begin{aligned}
q_4 &= v_x^B \cos q_3 - v_y^B \sin q_3 \\
q_5 &= v_x^B \sin q_3 + v_y^B \cos q_3
\end{aligned}$$

The turning parameters $\alpha$ and $x_\text{ICR}$ are grip- and terrain-dependent; see `motion_model.md` for their definitions.

### Gyro-augmented variant (used in `elevation_node_ekf`)

In the EKF predict step the wheel-differential yaw rate is replaced by the base-frame IMU gyro rate, averaged over the inter-cloud window:

$$q_6 = \dot{\psi} = \bar{\omega}_z^\text{gyro}$$

$v_y^B$ is updated accordingly ($v_y^B = -x_\text{ICR}\,\bar{\omega}_z^\text{gyro}$); $v_x^B$ is unchanged. The Jacobian structure is identical — $\omega_z^\text{gyro}$ is an exogenous input, not a state, so $F$ retains the same sparsity. The xy translation is still taken from the `ForwardSimulator` rollout (which uses the wheel model internally); the within-step heading inconsistency is second-order over $\Delta t = 0.1\,\text{s}$ and is negligible.

**Rationale:** during turns the wheel-differential model underestimates rotation because the effective grip factor $\alpha$ is only an approximation; the gyro measures actual rotation regardless of slip.

---

## 3. System linearisation 

### Flat ground - analytical model

Assume $\theta = \phi = 0$ and uniform $\mu$. Under these conditions $\alpha$ and $x_\text{ICR}$ are constants (quasi-static normal loads reduce to the CoM weight distribution, independent of state), so $v_x^B$, $v_y^B$, and $q_6$ depend only on $\mathbf{u}$, not on $\mathbf{q}$. The only state that appears in any component of $f$ is $q_3 = \psi$, entering $f_4$ and $f_5$ through $R_z(\psi)$.

The state-transition Jacobian $F = \partial f / \partial \mathbf{q}$ evaluated at the current operating point $\mathbf{q}$ is:

$$F = \frac{\partial f}{\partial \mathbf{q}} = \begin{pmatrix} 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & -q_5 & 0 & 0 & 0 \\ 0 & 0 & \phantom{-}q_4 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 \end{pmatrix}$$

The two non-trivial entries follow from differentiating the flat-ground rotation:

$$\frac{\partial f_4}{\partial q_3} = -v_x^B \sin q_3 - v_y^B \cos q_3 = -q_5, \qquad \frac{\partial f_5}{\partial q_3} = v_x^B \cos q_3 - v_y^B \sin q_3 = q_4$$

Both equal the current world-frame velocity components, so $F$ requires no trigonometric evaluation at runtime — only the current state. The linearised dynamics are:

$$\delta\dot{\mathbf{q}} \approx F\,\delta\mathbf{q}$$

### Nonflat ground - numerical linearization
On terrain, $F_t$ is computed numerically via central differences — three additional one-step `ForwardSimulator` rollouts, one per state dimension with perturbation $\delta$:

$$F_t[:,\, j] \approx \frac{f(\mathbf{x}_t + \delta\,\mathbf{e}_j,\, \boldsymbol{\omega}_t) - f(\mathbf{x}_t - \delta\,\mathbf{e}_j,\, \boldsymbol{\omega}_t)}{2\delta}$$
