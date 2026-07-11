"""Contract tests for the /cmd_joints command conditioning (pass-through, rear, slew, clamp)."""
import numpy as np

from helhest.control.command import condition_command
from helhest.control.command import JOINT_NAMES

Z = np.zeros(3, np.float32)


def test_joint_order():
    assert JOINT_NAMES == ("left_wheel_j", "rear_wheel_j", "right_wheel_j")


def test_forward_all_positive():
    # /cmd_joints input convention: forward = all wheels positive (no sign flip).
    cmd = condition_command(2.0, 2.0, Z, max_omega=4.0, max_slew=1e6, dt=0.1)
    assert cmd[0] > 0 and cmd[2] > 0
    np.testing.assert_allclose(cmd, [2.0, 2.0, 2.0], atol=1e-5)


def test_rear_is_mean():
    cmd = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1)
    np.testing.assert_allclose(cmd, [1.0, 2.0, 3.0], atol=1e-5)  # rear = mean(1,3) = 2


def test_slew_limit():
    # target [4,4,4] from rest, but max_slew*dt = 10*0.1 = 1 -> clipped to +1.
    cmd = condition_command(4.0, 4.0, Z, max_omega=4.0, max_slew=10.0, dt=0.1)
    np.testing.assert_allclose(cmd, [1.0, 1.0, 1.0], atol=1e-5)


def test_magnitude_clamp():
    # huge command, prev already near target so slew doesn't bind -> magnitude clamp to max_omega.
    prev = np.array([9.0, 9.0, 9.0], np.float32)
    cmd = condition_command(10.0, 10.0, prev, max_omega=4.0, max_slew=1e6, dt=0.1)
    np.testing.assert_allclose(cmd, [4.0, 4.0, 4.0], atol=1e-5)


def test_stop_ramps_down():
    prev = np.array([3.0, 3.0, 3.0], np.float32)
    cmd = condition_command(0.0, 0.0, prev, max_omega=4.0, max_slew=10.0, dt=0.1)
    # toward zero, but only by max_slew*dt = 1 per joint
    np.testing.assert_allclose(cmd, [2.0, 2.0, 2.0], atol=1e-5)


def test_turn_boost_amplifies_diff_keeps_mean():
    # boost=2 doubles the (wr-wl) differential but leaves the forward mean (rear) unchanged.
    base = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1)          # [1, 2, 3]
    boosted = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1, turn_boost=2.0)
    np.testing.assert_allclose(boosted, [0.0, 2.0, 4.0], atol=1e-5)  # mean 2 kept, diff 2 -> 4
    assert boosted[1] == base[1]  # rear (forward) unchanged


def test_turn_boost_default_is_noop():
    a = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1)
    b = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1, turn_boost=1.0)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_turn_direction():
    # planner turn: wr > wl -> intended +yaw (left/CCW). Right wheel commanded faster than left.
    cmd = condition_command(1.0, 2.0, Z, max_omega=4.0, max_slew=1e6, dt=0.1)
    assert cmd[2] > cmd[0]  # right faster than left -> left/CCW turn
