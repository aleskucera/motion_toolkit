"""Ouster beam geometry → LiDAR beam directions.

Two sensor shapes are supported:

* Cylindrical (OS0/OS1/OS2): channels at fixed altitude angles swept over azimuth
  columns — `ouster_beam_directions`.
* OSDome: a hemispherical fan pointing along one axis (the dome bulges that way).
  Parametrised by a polar angle from the dome axis (the "rings") swept over
  columns around it — `osdome_beam_directions`.

Ouster beams are individually calibrated; the exact table lives in the sensor's
metadata JSON (`beam_altitude_angles` / `beam_azimuth_angles`), read by
`load_ouster_metadata`. The nominal helpers are spec stand-ins.

All functions return (n_beams, 3) unit directions in the sensor frame
(x forward, y left, z up), ready for `PrimitiveLidar`.
"""

from __future__ import annotations

import json
import math

import numpy as np
import warp as wp

from ..sensor import LidarSensorConfig
from .lidar import GroundSpec
from .lidar import PrimitiveLidar

# Maps the canonical dome pole (+z) onto the chosen facing axis.
_DOME_AXES = {"front": "+x", "up": "+z", "down": "-z"}

# Ouster OSDome (Rev7, FW 3.1.x) datasheet specs.
# Source: https://data.ouster.io/downloads/datasheets/datasheet-rev7-v3p1-osdome.pdf
OSDOME_VERTICAL_FOV_DEG = 180.0  # hemispherical
OSDOME_CHANNELS = 128  # (also 32 / 64 variants)
OSDOME_MIN_RANGE_M = 0.5  # default (0.0 or 0.3 configurable)
OSDOME_MAX_RANGE_M = 45.0  # @ 80% reflectivity; ~20 m @ 10%
# Range precision (1σ) grows with distance: datasheet ≈ ±1 cm near → ±10 cm at
# 20 m (10% Lambertian). Modelled as base + quad·r², capped at the ±10 cm max.
OSDOME_RANGE_NOISE_BASE_M = 0.01
OSDOME_RANGE_NOISE_QUAD_M = 2.25e-4  # 0.01 + 2.25e-4·20² ≈ 0.10 m at 20 m
OSDOME_RANGE_NOISE_MAX_M = 0.10


def ouster_beam_directions(
    altitude_deg: np.ndarray,
    n_cols: int,
    azimuth_offset_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Cylindrical sensor: `channels` altitudes swept over `n_cols` azimuth columns.

    `altitude_deg` (C,) is per-channel elevation; `azimuth_offset_deg` (C,) the
    optional per-channel azimuth correction. Returns (C * n_cols, 3), channel-major.
    """
    alt = np.deg2rad(np.asarray(altitude_deg, dtype=np.float64))
    col = np.deg2rad(np.linspace(0.0, 360.0, n_cols, endpoint=False))
    off = np.zeros_like(alt) if azimuth_offset_deg is None else np.deg2rad(azimuth_offset_deg)
    az = col[None, :] + off[:, None]
    ca, sa = np.cos(alt)[:, None], np.sin(alt)[:, None]
    dirs = np.stack([ca * np.cos(az), ca * np.sin(az), np.broadcast_to(sa, az.shape)], axis=-1)
    return dirs.reshape(-1, 3).astype(np.float32)


def osdome_beam_directions(
    polar_deg: np.ndarray, n_cols: int, *, facing: str = "front"
) -> np.ndarray:
    """Hemispherical OSDome fan: `rings` polar angles swept over `n_cols` columns.

    `polar_deg` (R,) is the angle of each ring from the dome axis (0 = straight
    along the axis, 90 = the dome rim). Columns sweep 360° around that axis.
    `facing` orients the dome axis: 'front' (+x, the sensor looks ahead), 'up'
    (+z), or 'down' (−z). Returns (R * n_cols, 3), ring-major.
    """
    if facing not in _DOME_AXES:
        raise ValueError(f"facing must be one of {sorted(_DOME_AXES)}; got {facing!r}")
    theta = np.deg2rad(np.asarray(polar_deg, dtype=np.float64))  # from the dome axis
    phi = np.deg2rad(np.linspace(0.0, 360.0, n_cols, endpoint=False))  # around it
    st = np.sin(theta)[:, None]
    ct = np.broadcast_to(np.cos(theta)[:, None], (len(theta), n_cols))
    # Canonical frame: dome axis = +z, ring opens in the xy-plane.
    x = st * np.cos(phi)[None, :]
    y = st * np.sin(phi)[None, :]
    z = ct
    if facing == "up":
        dirs = np.stack([x, y, z], axis=-1)
    elif facing == "down":
        dirs = np.stack([x, y, -z], axis=-1)
    else:  # front: rotate +z → +x (R_y(90°): [x,y,z] → [z, y, -x])
        dirs = np.stack([z, y, -x], axis=-1)
    return dirs.reshape(-1, 3).astype(np.float32)


def osdome_sensor_config(columns: int = 1024, channels: int = OSDOME_CHANNELS) -> LidarSensorConfig:
    """`LidarSensorConfig` for a FRONT-facing OSDome, from the Rev7 datasheet.

    The dome's 180° hemisphere, pointing along +x, spans the full vertical range
    and half the azimuth in the range-image frame, so `el_fov = ±90°` and
    `az_fov = ±90°`. `columns` selects the resolution mode (512 / 1024 / 2048).
    """
    return LidarSensorConfig(
        el_fov_deg=(-90.0, 90.0),
        channels=channels,
        columns=columns,
        az_fov_deg=(-90.0, 90.0),
        min_range_m=OSDOME_MIN_RANGE_M,
        max_range_m=OSDOME_MAX_RANGE_M,
        range_noise_base_m=OSDOME_RANGE_NOISE_BASE_M,
        range_noise_quad_m=OSDOME_RANGE_NOISE_QUAD_M,
        range_noise_max_m=OSDOME_RANGE_NOISE_MAX_M,
    )


def nominal_osdome_polar(rings: int = 128) -> np.ndarray:
    """Nominal OSDome ring angles: uniform over the half-FOV from the dome axis.

    128 rings over 0°→90° gives ~0.7° spacing, matching the datasheet's "up to
    0.7° vertical angular resolution". Starts just off the pole so columns don't
    all collapse onto the axis. Prefer the real per-beam table from
    `load_ouster_metadata` when available.
    """
    return np.linspace(0.5, OSDOME_VERTICAL_FOV_DEG / 2.0, rings)


def make_osdome_lidar(
    ground: GroundSpec,
    *,
    sensor: LidarSensorConfig | None = None,
    channels: int = OSDOME_CHANNELS,
    cols: int = 1024,
    facing: str = "front",
    dropout: float = 0.0,
    device: wp.context.Device | None = None,
) -> PrimitiveLidar:
    """A `PrimitiveLidar` configured to the OSDome datasheet (Rev7).

    Nominal uniform hemisphere beams + the datasheet range window and precision
    curve, both taken from a `LidarSensorConfig`. Pass `sensor` to share one
    config with the perception filter; otherwise it's built from `channels`/`cols`.
    `dropout` isn't a datasheet figure (OSDome returns are dense) — a small
    optional stand-in for the far / low-reflectivity returns a reflectivity model
    would drop.
    """
    if sensor is None:
        sensor = osdome_sensor_config(columns=cols, channels=channels)
    dirs = osdome_beam_directions(
        nominal_osdome_polar(sensor.channels), sensor.columns, facing=facing
    )
    return PrimitiveLidar.from_sensor(sensor, dirs, ground=ground, dropout=dropout, device=device)


def _find(meta: dict, key: str):
    """Fetch `key` from the metadata, tolerating firmware layout differences."""
    if key in meta:
        return meta[key]
    for sub in ("beam_intrinsics", "lidar_intrinsics"):
        if isinstance(meta.get(sub), dict) and key in meta[sub]:
            return meta[sub][key]
    return None


def load_ouster_metadata(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse an Ouster metadata JSON → (beam_altitude_deg, beam_azimuth_deg, n_cols).

    Handles legacy (top-level) and 2.x (`beam_intrinsics`) layouts and reads the
    column count from `lidar_data_format` or `lidar_mode`.
    """
    with open(path) as f:
        meta = json.load(f)

    alt = _find(meta, "beam_altitude_angles")
    if alt is None:
        raise ValueError(f"no beam_altitude_angles in {path}")
    alt = np.asarray(alt, dtype=np.float64)
    az = _find(meta, "beam_azimuth_angles")
    az = np.zeros_like(alt) if az is None else np.asarray(az, dtype=np.float64)

    n_cols = 1024
    fmt = meta.get("lidar_data_format")
    if isinstance(fmt, dict) and "columns_per_frame" in fmt:
        n_cols = int(fmt["columns_per_frame"])
    else:
        mode = meta.get("lidar_mode") or _find(meta, "lidar_mode")
        if isinstance(mode, str) and "x" in mode:  # e.g. "1024x10"
            n_cols = int(mode.split("x")[0])

    return alt, az, n_cols


def sensor_config_from_ouster_metadata(
    path: str,
    *,
    min_range_m: float = 0.0,
    max_range_m: float = math.inf,
    range_noise_base_m: float = 0.0,
    range_noise_quad_m: float = 0.0,
    range_noise_max_m: float = math.inf,
) -> LidarSensorConfig:
    """Build a `LidarSensorConfig` for the real sensor from its metadata JSON.

    Geometry (vertical FOV from the calibrated `beam_altitude_angles`, channels,
    and columns) comes from the metadata; the range window and precision curve
    are datasheet figures that metadata omits, so pass them from the datasheet.
    Lives here (not on `LidarSensorConfig`) to keep that contract free of any
    Ouster/JSON specifics.
    """
    altitudes, _azimuths, n_cols = load_ouster_metadata(path)
    return LidarSensorConfig(
        el_fov_deg=(float(altitudes.min()), float(altitudes.max())),
        channels=int(len(altitudes)),
        columns=int(n_cols),
        az_fov_deg=(-180.0, 180.0),
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        range_noise_base_m=range_noise_base_m,
        range_noise_quad_m=range_noise_quad_m,
        range_noise_max_m=range_noise_max_m,
    )
