"""LidarSensorConfig validation + the OSDome datasheet preset.

Run: python tests/test_sensor.py
"""

from __future__ import annotations

import json
import math
import os
import tempfile

from helhest.perception import DynamicFilterConfig
from helhest.perception import LidarSensorConfig
from helhest.perception.sim import osdome_sensor_config
from helhest.perception.sim import sensor_config_from_ouster_metadata


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


def test_validation() -> None:
    # A valid config constructs fine.
    LidarSensorConfig(el_fov_deg=(-45.0, 45.0), channels=64, columns=1024)
    # Each guard rejects the obvious mistake.
    assert _raises(lambda: LidarSensorConfig(el_fov_deg=(45.0, -45.0), channels=64, columns=1024))
    assert _raises(lambda: LidarSensorConfig(el_fov_deg=(-45.0, 45.0), channels=0, columns=1024))
    assert _raises(
        lambda: LidarSensorConfig(
            el_fov_deg=(-45.0, 45.0), channels=64, columns=1024, az_fov_deg=(90.0, -90.0)
        )
    )
    assert _raises(
        lambda: LidarSensorConfig(
            el_fov_deg=(-45.0, 45.0), channels=64, columns=1024, min_range_m=50.0, max_range_m=45.0
        )
    )
    assert _raises(
        lambda: LidarSensorConfig(
            el_fov_deg=(-45.0, 45.0), channels=64, columns=1024, range_noise_base_m=-0.1
        )
    )


def test_osdome_preset() -> None:
    cfg = osdome_sensor_config(columns=2048)
    assert cfg.el_fov_deg == (-90.0, 90.0) and cfg.az_fov_deg == (-90.0, 90.0)
    assert cfg.channels == 128 and cfg.columns == 2048
    assert cfg.min_range_m == 0.5 and cfg.max_range_m == 45.0
    assert cfg.range_noise_max_m == 0.10 and not math.isinf(cfg.range_noise_max_m)


def test_filter_config_from_sensor() -> None:
    sensor = osdome_sensor_config(columns=1024)
    cfg = DynamicFilterConfig.from_sensor(sensor, margin_m=0.4, margin_rel=0.05)
    # FOV / resolution / min-range come from the sensor; margins from the call.
    assert cfg.el_min_deg == -90.0 and cfg.el_max_deg == 90.0
    assert cfg.az_bins == 1024 and cfg.el_bins == 128  # sensor columns / channels
    assert cfg.min_range_m == sensor.min_range_m
    assert cfg.margin_m == 0.4 and cfg.margin_rel == 0.05
    # Bin counts are overridable.
    assert DynamicFilterConfig.from_sensor(sensor, el_bins=180).el_bins == 180


def test_from_ouster_metadata() -> None:
    # A minimal 2.x-style metadata blob: 4 beams over ±10°, 2048 columns.
    meta = {
        "beam_intrinsics": {
            "beam_altitude_angles": [-10.0, -3.0, 3.0, 10.0],
            "beam_azimuth_angles": [0.0, 0.0, 0.0, 0.0],
        },
        "lidar_data_format": {"columns_per_frame": 2048},
    }
    path = os.path.join(tempfile.mkdtemp(), "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f)

    cfg = sensor_config_from_ouster_metadata(path, min_range_m=0.5, max_range_m=120.0)
    assert cfg.el_fov_deg == (-10.0, 10.0)  # from the calibrated altitude table
    assert cfg.channels == 4 and cfg.columns == 2048
    assert cfg.min_range_m == 0.5 and cfg.max_range_m == 120.0


def main() -> None:
    test_validation()
    test_osdome_preset()
    test_filter_config_from_sensor()
    test_from_ouster_metadata()
    print("PASS: LidarSensorConfig + OSDome preset + from_sensor + from_metadata (4 tests)")


if __name__ == "__main__":
    main()
