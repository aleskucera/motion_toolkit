"""Contract tests for the /cmd_joints command conditioning (sign flip, rear, slew, clamp)."""
import numpy as np

from helhest.control.command import condition_command
from helhest.control.command import JOINT_NAMES

Z = np.zeros(3, np.float32)


def test_joint_order():
    assert JOINT_NAMES == ("left_wheel_j", "rear_wheel_j", "right_wheel_j")


def test_forward_sign_flip():
    # planner "forward" = both wheels positive -> real robot: left/rear negative, right positive.
    cmd = condition_command(2.0, 2.0, Z, max_omega=4.0, max_slew=1e6, dt=0.1)
    assert cmd[0] < 0 < cmd[2]  # left backward-sign, right forward-sign
    np.testing.assert_allclose(cmd, [-2.0, -2.0, 2.0], atol=1e-5)


def test_rear_is_mean():
    cmd = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1)
    np.testing.assert_allclose(cmd, [-1.0, -2.0, 3.0], atol=1e-5)  # rear = -mean(1,3) = -2


def test_slew_limit():
    # target [-4,-4,4] from rest, but max_slew*dt = 10*0.1 = 1 -> clipped to +-1.
    cmd = condition_command(4.0, 4.0, Z, max_omega=4.0, max_slew=10.0, dt=0.1)
    np.testing.assert_allclose(cmd, [-1.0, -1.0, 1.0], atol=1e-5)


def test_magnitude_clamp():
    # huge command, prev already near target so slew doesn't bind -> magnitude clamp to max_omega.
    prev = np.array([-9.0, -9.0, 9.0], np.float32)
    cmd = condition_command(10.0, 10.0, prev, max_omega=4.0, max_slew=1e6, dt=0.1)
    np.testing.assert_allclose(cmd, [-4.0, -4.0, 4.0], atol=1e-5)


def test_stop_ramps_down():
    prev = np.array([-3.0, -3.0, 3.0], np.float32)
    cmd = condition_command(0.0, 0.0, prev, max_omega=4.0, max_slew=10.0, dt=0.1)
    # toward zero, but only by max_slew*dt = 1 per joint
    np.testing.assert_allclose(cmd, [-2.0, -2.0, 2.0], atol=1e-5)


def test_turn_direction():
    # planner turn: wr > wl -> intended +yaw (model). Left less negative than right is positive.
    cmd = condition_command(1.0, 2.0, Z, max_omega=4.0, max_slew=1e6, dt=0.1)
    assert cmd[2] > -cmd[0]  # |right| > |left| -> real yaw term (wR+wL in real conv) nonzero
