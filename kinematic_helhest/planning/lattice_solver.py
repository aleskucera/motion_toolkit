"""Orientation-aware cost-to-go V(row, col, heading), value-iterated on the GPU.

Engine-independent: takes the per-pose `blocked` + graded-`tilt` fields a feasibility producer makes
(here CostToGo's settle) plus a goal cell, and value-iterates over a (row, col, heading) lattice with
forward-arc motion primitives capped at a minimum turn radius and swept-collision checked. A pose from
which the goal is unreachable for a forward-only robot keeps cost +inf -- so sampling V at the robot's
(x, y, yaw) penalises exactly the misaligned approaches a 2D geodesic can't express. The convergence
loop runs ON DEVICE (capture_while), so the whole solve is CUDA-graph-capturable.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp


@wp.kernel
def _free_goal_kernel(blocked: wp.array(dtype=wp.float32, ndim=3), goal_rc: wp.array(dtype=wp.int32)):
    """Force the goal cell feasible at every heading, so the value iteration always has a source."""
    t = wp.tid()
    blocked[goal_rc[0], goal_rc[1], t] = 0.0


@wp.kernel
def _init_lattice_kernel(
    goal_rc: wp.array(dtype=wp.int32),  # [2] (row, col), computed on device
    inf: wp.float32,
    dist: wp.array(dtype=wp.float32, ndim=3),
):
    """Seed: 0 at the goal cell (every heading), +inf everywhere else."""
    r, c, t = wp.tid()
    if r == goal_rc[0] and c == goal_rc[1]:
        dist[r, c, t] = 0.0
    else:
        dist[r, c, t] = inf


@wp.kernel
def _keep_going_kernel(
    changed: wp.array(dtype=wp.int32),
    iter_count: wp.array(dtype=wp.int32),
    cap: wp.int32,
    keep_running: wp.array(dtype=wp.int32),
):
    """Device-side loop condition (so capture_while keeps the value iteration on the GPU, no host
    sync): keep going while the last body improved SOME cell and we're under the cap. dim=1."""
    it = iter_count[0] + 1
    iter_count[0] = it
    if changed[0] != 0 and it < cap:
        keep_running[0] = 1
    else:
        keep_running[0] = 0


@wp.kernel
def _relax_lattice_pose_kernel(
    dist_in: wp.array(dtype=wp.float32, ndim=3),
    blocked: wp.array(
        dtype=wp.float32, ndim=3
    ),  # [h, w, n_theta] PER-POSE feasibility (1 = blocked)
    tilt: wp.array(
        dtype=wp.float32, ndim=3
    ),  # [h, w, n_theta] per-pose body tilt (rad), graded cost
    prim_dr: wp.array(dtype=wp.int32, ndim=2),  # [n_theta, n_prim] endpoint row offset
    prim_dc: wp.array(dtype=wp.int32, ndim=2),  # endpoint col offset
    prim_heading: wp.array(dtype=wp.int32, ndim=2),  # heading bin the arc ends at
    prim_cost: wp.array(dtype=wp.float32, ndim=2),  # arc length
    sweep_dr: wp.array(dtype=wp.int32, ndim=3),  # [n_theta, n_prim, max_sweep] swept-cell offsets
    sweep_dc: wp.array(dtype=wp.int32, ndim=3),
    sweep_n: wp.array(dtype=wp.int32, ndim=2),  # [n_theta, n_prim] swept-cell count
    n_prim: wp.int32,
    tilt_weight: wp.float32,  # 0 -> pure distance; >0 -> prefer low-tilt poses
    inf: wp.float32,
    dist_out: wp.array(dtype=wp.float32, ndim=3),
    changed: wp.array(dtype=wp.int32),
):
    """One min-relaxation sweep of the (row, col, heading) value function:

        V_out[s] = min( V_in[s], min over forward-arc primitives p
                         of  cost(p) + V_in[ next(s, p) ]   if p's swept cells are all free )

    cost(p) = arc_length * (1 + tilt_weight * mean tilt over the arc's swept cells), so with
    tilt_weight > 0 the geodesic PREFERS flatter poses (not just avoids blocked ones). Feasibility and
    graded cost come from the PER-POSE field (the robot's settle), sampled at the swept cells using the
    pose's own heading t (the arc rotates little over one step), so a wall face -- where the body tilts/
    high-centers -- blocks the crossing arc while flat ground stays cheap. The whole swept arc must be
    clear (not just the endpoint), so the robot can't jump a thin wall. Iterating to a fixed point gives
    the forward-only cost-to-go; a misaligned pose from which the goal is unreachable stays +inf."""
    r, c, t = wp.tid()
    h = dist_in.shape[0]
    w = dist_in.shape[1]
    if blocked[r, c, t] > 0.5:
        dist_out[r, c, t] = inf
        return
    best = dist_in[r, c, t]
    for p in range(n_prim):
        ok = int(1)
        ns = sweep_n[t, p]
        tsum = float(0.0)
        for s in range(ns):
            sr = r + sweep_dr[t, p, s]
            sc = c + sweep_dc[t, p, s]
            inb = int(0)
            if sr >= 0 and sr < h and sc >= 0 and sc < w:
                inb = 1
            scr = wp.clamp(sr, 0, h - 1)
            scc = wp.clamp(sc, 0, w - 1)
            if inb == 0 or blocked[scr, scc, t] > 0.5:
                ok = 0
            tsum += tilt[scr, scc, t]
        if ok == 1:
            nr = r + prim_dr[t, p]
            nc = c + prim_dc[t, p]
            if nr >= 0 and nr < h and nc >= 0 and nc < w:
                arc = prim_cost[t, p]
                if ns > 0:
                    arc = arc * (1.0 + tilt_weight * tsum / float(ns))
                best = wp.min(best, arc + dist_in[nr, nc, prim_heading[t, p]])
    dist_out[r, c, t] = best
    if best < dist_in[r, c, t]:
        changed[0] = 1


def _build_primitives(n_theta, resolution, step, turn_radius, max_sweep, nseg):
    """Host-side forward-arc motion primitives. For each heading bin and each turn rate, integrate
    the arc of length `step`, and record: endpoint cell offset (dr, dc), resulting heading bin, arc
    cost, and the swept cells (offsets) the arc passes through (for collision). Min turn radius
    `turn_radius` caps the turn rate, so turning costs space -- the whole point."""
    dth = 2.0 * math.pi / n_theta
    turns = [
        -step / turn_radius,
        -step / turn_radius / 2.0,
        0.0,
        step / turn_radius / 2.0,
        step / turn_radius,
    ]  # dtheta over the step
    n_prim = len(turns)
    prim_dr = np.zeros((n_theta, n_prim), np.int32)
    prim_dc = np.zeros((n_theta, n_prim), np.int32)
    prim_heading = np.zeros((n_theta, n_prim), np.int32)
    prim_cost = np.full((n_theta, n_prim), step, np.float32)
    sweep_dr = np.zeros((n_theta, n_prim, max_sweep), np.int32)
    sweep_dc = np.zeros((n_theta, n_prim, max_sweep), np.int32)
    sweep_n = np.zeros((n_theta, n_prim), np.int32)
    for it in range(n_theta):
        th = (it + 0.5) * dth
        for p, dth_p in enumerate(turns):
            x, y = 0.0, 0.0
            cells = []
            for s in range(1, nseg):
                cth = th + dth_p * (float(s) - 0.5) / float(nseg - 1)
                x += (step / float(nseg - 1)) * math.cos(cth)
                y += (step / float(nseg - 1)) * math.sin(cth)
                cells.append((int(round(y / resolution)), int(round(x / resolution))))
            prim_dc[it, p] = int(round(x / resolution))
            prim_dr[it, p] = int(round(y / resolution))
            prim_heading[it, p] = int(math.floor(((th + dth_p) % (2.0 * math.pi)) / dth)) % n_theta
            uniq = sorted(set(cells))[:max_sweep]
            for s, (cr, cc) in enumerate(uniq):
                sweep_dr[it, p, s] = cr
                sweep_dc[it, p, s] = cc
            sweep_n[it, p] = len(uniq)
    return n_prim, prim_dr, prim_dc, prim_heading, prim_cost, sweep_dr, sweep_dc, sweep_n


class LatticeValueSolver:
    def __init__(
        self,
        resolution: float,
        height: int,
        width: int,
        n_theta: int = 16,
        turn_radius: float = 0.6,
        step: float | None = None,
        device: wp.Device | None = None,
    ):
        self.resolution = resolution
        self.height = height
        self.width = width
        self.n_theta = n_theta
        self.device = wp.get_device(device)
        self._inf = 1.0e30
        self._step = float(step) if step is not None else 2.0 * self.resolution

        # a single arc can sweep ~step/resolution cells; size the swept-cell buffer + arc sampling
        # to that ratio so fine grids don't truncate the collision check and jump thin walls.
        step_cells = self._step / self.resolution
        max_sweep = max(6, int(math.ceil(step_cells)) * 2 + 3)
        nseg = max(8, int(step_cells * 4))
        n_prim, prim_dr, prim_dc, prim_heading, prim_cost, sweep_dr, sweep_dc, sweep_n = _build_primitives(
            self.n_theta, self.resolution, self._step, float(turn_radius), max_sweep, nseg
        )
        self.n_prim = n_prim
        # motion-primitive table on device, indexed [heading_bin, primitive]: where each forward arc
        # lands + what it crosses (see _build_primitives). The relax kernel reads these every sweep.
        with wp.ScopedDevice(self.device):
            self._prim_dr = wp.array(prim_dr, dtype=wp.int32)  # endpoint row offset of the arc
            self._prim_dc = wp.array(prim_dc, dtype=wp.int32)  # endpoint col offset
            self._prim_heading = wp.array(prim_heading, dtype=wp.int32)  # heading bin the arc ends at
            self._prim_cost = wp.array(prim_cost, dtype=wp.float32)  # arc length (the move's base cost)
            self._sweep_dr = wp.array(sweep_dr, dtype=wp.int32)  # row offsets of the cells the arc crosses
            self._sweep_dc = wp.array(sweep_dc, dtype=wp.int32)  # col offsets of those swept cells
            self._sweep_n = wp.array(sweep_n, dtype=wp.int32)  # how many swept cells each arc has
            # two value buffers, ping-ponged each sweep (read one, write the other, swap); +changed flag
            self._dist_a = wp.zeros((self.height, self.width, self.n_theta), dtype=wp.float32)
            self._dist_b = wp.zeros((self.height, self.width, self.n_theta), dtype=wp.float32)
            self._changed = wp.zeros(1, dtype=wp.int32)  # >0 if any cell improved this sweep (convergence)
            self._keep_running = wp.zeros(1, dtype=wp.int32)  # device while-condition for capture_while
            self._iter = wp.zeros(1, dtype=wp.int32)
        self._cap = self.height + self.width  # max bodies (each = 2 sweeps -> 2*(h+w) sweeps total)

    def _relax(self, dist_in, dist_out, blocked, tilt, tilt_weight):
        """One min-relaxation sweep dist_in -> dist_out (race-free pull); raises self._changed if any
        cell improved."""
        wp.launch(
            _relax_lattice_pose_kernel,
            dim=(self.height, self.width, self.n_theta),
            inputs=[
                dist_in,
                blocked,
                tilt,
                self._prim_dr,
                self._prim_dc,
                self._prim_heading,
                self._prim_cost,
                self._sweep_dr,
                self._sweep_dc,
                self._sweep_n,
                self.n_prim,
                float(tilt_weight),
                self._inf,
            ],
            outputs=[dist_out, self._changed],
            device=self.device,
        )

    def _record_solve(self, blocked, tilt, goal_rc, tilt_weight, capture):
        """Record the device-side value iteration: free the goal cell, seed, then min-relax to a fixed
        point. The convergence loop runs ON DEVICE via capture_while (graph-safe) -- a 2-SWEEP body
        (a->b->a) keeps the ping-pong buffers fixed inside the captured graph, so the result always
        lands in self._dist_a. capture=False uses an eager host while-loop (CPU / no-graph fallback)."""
        dev = self.device
        grid_dim = (self.height, self.width, self.n_theta)
        wp.launch(_free_goal_kernel, dim=self.n_theta, inputs=[blocked, goal_rc], device=dev)
        wp.launch(
            _init_lattice_kernel,
            dim=grid_dim,
            inputs=[goal_rc, self._inf],
            outputs=[self._dist_a],
            device=dev,
        )
        self._keep_running.fill_(1)
        self._iter.zero_()

        def body():
            self._changed.zero_()
            self._relax(self._dist_a, self._dist_b, blocked, tilt, tilt_weight)
            self._relax(self._dist_b, self._dist_a, blocked, tilt, tilt_weight)
            wp.launch(
                _keep_going_kernel,
                dim=1,
                inputs=[self._changed, self._iter, self._cap, self._keep_running],
                device=dev,
            )

        if capture:
            wp.capture_while(self._keep_running, body)
        else:
            while True:
                body()
                if int(self._keep_running.numpy()[0]) == 0:
                    break
        return self._dist_a
