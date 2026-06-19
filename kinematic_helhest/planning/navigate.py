"""Robot-centered rolling-map MPPI navigation (Phase 2).

Each cycle: ingest a robot-centered local device map, replan an MPPI horizon to the
goal in the robot frame (robot at local (0,0,0)), execute the first control on the
true robot, shift, repeat. Written against the *generic* perception contract
(`.elevation` device array + `.resolution` + `.bounds`) — NOT terrain_toolkit — so
the synthetic stand-in (synthetic_perception.crop_window) and the real pipeline are
interchangeable. The world bookkeeping (odom, global goal) lives here.

For the synthetic demo the "true robot" is the kinematic engine stepping on the full
world map (a B=1,T=1 rollout — the model is memoryless, so one rollout step == one
real step).

Demo:  python -m kinematic_helhest.planning.navigate [--device cuda] [--gx 4 --gy 1.5]
"""
import argparse
from dataclasses import dataclass

import numpy as np
import warp as wp

from .. import heightmap as hmmod
from ..engine import GridParams
from ..engine import RobotParams
from ..engine import Simulator
from ..engine import SolverParams
from .mppi import _to_omega
from .mppi_gpu import MppiGpu
from .synthetic_perception import crop_window
from .synthetic_perception import to_local

_W = dict(term=3.0, run=0.3, invalid=1e5, eff=2e-3, smooth=2e-3)


@dataclass
class NavConfig:
    half_extent: float = 8.0   # local window half-size [m] (must cover the T=100 horizon reach)
    res: float = 0.06          # local window cell size [m]
    T: int = 100               # MPPI horizon (5 s lookahead; under the 50 ms control budget)
    B: int = 2048              # MPPI samples
    dt: float = 0.05
    n_refine: int = 3
    sigma: float = 0.5         # per-step jitter std (local variation)
    sigma_bias: float = 1.0    # sustained per-rollout bias std (broad spatial fan)
    lam: float = 0.5
    wmax: float = 4.0
    clear_margin: float = 0.05
    resid_tol: float = 1e-2
    goal_tol: float = 0.3
    mu_value: float = 0.8
    # Traversability: penalize the robot's total tilt from vertical along the rollout,
    # but only ABOVE tilt_free (gentle drivable ramps are free, so slope goals stay
    # reachable). Obstacle avoidance comes from robust control, not a map.
    tilt_w: float = 300.0      # MPPI weight on mean max(tilt - tilt_free, 0)^2
    tilt_free_deg: float = 12.0  # tilt below this is free [deg]


class Navigator:
    """MPPI planner over a rolling robot-centered local map. The robot is at local
    (0,0,0) every cycle; the goal is supplied already projected into the local frame."""

    def __init__(self, cfg, device="cpu", robot_params=None, seed=0):
        self.cfg = cfg
        self.dev = device
        self.rp = robot_params or RobotParams()
        self.params = SolverParams(dt=cfg.dt, k_turn=2.0, newton_iters=12, atol=1e-4)  # forward-only: loose settle
        self.seed = seed
        self.sim = None  # built lazily on the first map (grid sizes from it)
        self.drv = None  # MppiGpu, built with the sim

        # Window must cover the worst-case horizon reach, else rollouts sample
        # off-grid (clamped -> wrong). v_max = wheel_radius * wmax.
        reach = self.rp.wheel_radius * cfg.wmax * cfg.T * cfg.dt
        margin = self.rp.rear_offset
        assert cfg.half_extent > reach + margin, (
            f"half_extent {cfg.half_extent} must exceed horizon reach {reach:.2f} + "
            f"margin {margin:.2f}; raise half_extent or lower T/wmax")

    def replan(self, local_map, goal_local, U):
        """local_map: DeviceMap-shaped (.elevation/.resolution/.bounds). Returns the
        updated nominal control U [T,2] and the predicted local path xy [T+1,2]."""
        cfg = self.cfg
        # one cell-center convention end-to-end: the grid origin IS the bounds min corner.
        x0, y0 = local_map.bounds[0], local_map.bounds[2]
        raw_H, cell = local_map.elevation, local_map.resolution
        if self.sim is None:  # fixed window dims -> build the preallocated sim + driver once
            ny, nx = raw_H.shape
            grid = GridParams(nx, ny, cell, x0, y0)
            self.sim = Simulator(self.rp, self.params, grid, cfg.B, cfg.T, self.dev)
            w = dict(_W, tilt=cfg.tilt_w, tilt_free=np.radians(cfg.tilt_free_deg))
            self.drv = MppiGpu(self.sim, cfg.sigma, cfg.lam, cfg.wmax, w,
                               cfg.clear_margin, cfg.resid_tol, self.seed, sigma_bias=cfg.sigma_bias)
        self.sim.set_terrain(raw_H)            # borrow + dilate, no alloc
        self.sim.set_uniform_friction(cfg.mu_value)

        state = (0.0, 0.0, 0.0)  # robot at local origin
        self.drv.set_nominal(U)
        self.drv.replan(state, goal_local, cfg.n_refine)   # whole MPPI refine on GPU
        U = self.drv.nominal()
        controlled, _, _, _ = self.sim.rollout(_to_omega(np.tile(U, (cfg.B, 1, 1))), state)
        return U, controlled[:, 0, :2].copy()


class WorldRobot:
    """Ground-truth robot for the synthetic demo: the engine stepping on the full
    world map (B=1, T=1)."""

    def __init__(self, world_hm, cfg, device="cpu", robot_params=None):
        nx, ny, c = world_hm.nx, world_hm.ny, world_hm.cell
        mu = hmmod.Heightmap(np.full((ny, nx), cfg.mu_value), (world_hm.x0, world_hm.y0), c)
        params = SolverParams(dt=cfg.dt, k_turn=2.0, newton_iters=12)
        rp = robot_params or RobotParams()
        self.sim = Simulator(
            rp, params,
            GridParams(world_hm.nx, world_hm.ny, world_hm.cell, world_hm.x0, world_hm.y0),
            1, 1, device,
        )
        self.sim.set_terrain(wp.array(np.ascontiguousarray(world_hm.H, np.float32),
                                      dtype=wp.float32, device=device))
        self.sim.set_friction(mu)

    def step(self, state, ctrl):
        """Advance world `state` (x,y,yaw) by wheel control `ctrl` (wL, wR)."""
        Ub = np.asarray(ctrl[:2], np.float32).reshape(1, 1, 2)
        controlled, derived, clear, resid = self.sim.rollout(_to_omega(Ub), state)
        return controlled[1, 0].copy(), derived[1, 0].copy(), float(clear[0, 0]), float(resid[0, 0])


def drive(world_hm, start, goal, cfg, device="cpu", max_cycles=300, seed=0, record=False):
    """Receding-horizon rolling-map MPPI over a static world. Returns
    (world_path [K,2], reached bool, frames). `frames` non-empty only if record."""
    rp = RobotParams()
    nav = Navigator(cfg, device=device, robot_params=rp, seed=seed)
    truth = WorldRobot(world_hm, cfg, device=device, robot_params=rp)
    goal = np.asarray(goal[:2], np.float64)

    state = np.asarray(start, np.float32)        # world (x, y, yaw)
    U = np.full((cfg.T, 2), 1.5, np.float32)     # nominal wheel speeds
    path, frames = [state[:2].copy()], []
    for _ in range(max_cycles):
        if np.hypot(state[0] - goal[0], state[1] - goal[1]) < cfg.goal_tol:
            return np.array(path), True, frames
        local_map = crop_window(world_hm, state[:2], state[2], cfg.half_extent, cfg.res, device)
        goal_local = to_local(goal, state)
        U, plan_local = nav.replan(local_map, goal_local, U)
        if record:
            frames.append({"state": state.copy(), "plan_local": plan_local})
        state, _, _, _ = truth.step(state, U[0])
        path.append(state[:2].copy())
        U = np.roll(U, -1, axis=0)
        U[-1] = U[-2]
    return np.array(path), False, frames


def synth_lidar(world_hm, state, half_extent, rng, n=80000):
    """Synthetic robot-frame point cloud (the Phase-4 demo's stand-in lidar).

    Samples the world under a robot-centered window, in the robot frame with
    gravity-aligned z. On a real robot this is replaced by the actual sensor cloud.
    """
    lx = rng.uniform(-half_extent, half_extent, n)
    ly = rng.uniform(-half_extent, half_extent, n)
    c, s = np.cos(state[2]), np.sin(state[2])
    wx = state[0] + c * lx - s * ly
    wy = state[1] + s * lx + c * ly
    wz = world_hm.sample(wx, wy)
    return np.stack([lx, ly, wz], axis=1).astype(np.float32)


def drive_perception(world_hm, start, goal, cfg, device="cuda", max_cycles=300, seed=0):
    """Phase 4: the real terrain_toolkit pipeline in the loop (lazy-imported).

    Each cycle a synthetic lidar feeds robot-frame points to a `TerrainPipeline`;
    the resulting on-device `TerrainMapGPU` (.elevation/.resolution/.bounds — the
    same generic contract the synthetic stand-in produces) is consumed zero-copy by
    the rolling planner. Swapping `synth_lidar` for a real sensor is the only change
    needed on hardware.
    """
    try:
        from terrain_toolkit import TerrainPipeline
    except ImportError as e:  # optional dependency
        raise ImportError(
            "the --perception path needs terrain_toolkit; install the extra, e.g. "
            "`uv pip install -e ../terrain_toolkit --no-deps`") from e

    rp = RobotParams()
    nav = Navigator(cfg, device=device, robot_params=rp, seed=seed)
    truth = WorldRobot(world_hm, cfg, device=device, robot_params=rp)
    bounds = (-cfg.half_extent, cfg.half_extent, -cfg.half_extent, cfg.half_extent)
    pipe = TerrainPipeline(cfg.res, bounds, inpaint=True, device=device)
    rng = np.random.default_rng(seed + 1)
    goal = np.asarray(goal[:2], np.float64)

    state = np.asarray(start, np.float32)
    U = np.full((cfg.T, 2), 1.5, np.float32)
    path = [state[:2].copy()]
    for _ in range(max_cycles):
        if np.hypot(state[0] - goal[0], state[1] - goal[1]) < cfg.goal_tol:
            return np.array(path), True, []
        cloud = synth_lidar(world_hm, state, cfg.half_extent, rng)
        tmap = pipe.process(cloud, return_device=True)   # borrowed TerrainMapGPU
        U, _ = nav.replan(tmap, to_local(goal, state), U)
        state, _, _, _ = truth.step(state, U[0])
        path.append(state[:2].copy())
        U = np.roll(U, -1, axis=0)
        U[-1] = U[-2]
    return np.array(path), False, []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda", help="warp device: cpu or cuda")
    ap.add_argument("--gx", type=float, default=4.0)
    ap.add_argument("--gy", type=float, default=1.5)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--perception", action="store_true",
                    help="run the real terrain_toolkit pipeline (synthetic lidar) instead of crop_window")
    args = ap.parse_args()

    world = hmmod.demo_terrain()
    start = (0.0, 0.0, 0.0)
    goal = (args.gx, args.gy)
    cfg = NavConfig()
    drive_fn = drive_perception if args.perception else drive
    path, reached, _ = drive_fn(world, start, goal, cfg, device=args.device, max_cycles=args.cycles)
    d = float(np.hypot(path[-1, 0] - goal[0], path[-1, 1] - goal[1]))
    print(f"reached={reached}  final=({path[-1,0]:+.2f},{path[-1,1]:+.2f})  "
          f"dist_to_goal={d:.2f}m  cycles={len(path)-1}")


if __name__ == "__main__":
    main()
