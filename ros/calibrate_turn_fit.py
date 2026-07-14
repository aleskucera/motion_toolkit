"""Fit the skid-steer turn gain K_TURN from a calibration drive.

Ground truth for yaw is the IMU gyro (validated ~4% vs the tuned ICP localization), so the fit
needs NO node / GPU / container -- just the recorded bag. It auto-detects the wheel convention
(which of R+-L is the turn term) and which gyro axis is yaw, gates IMU spikes, then fits
    yaw_rate = R * turn_term / (2 * half_track * alpha),   alpha = 1 + K_TURN * mu
and reports K_TURN = (alpha - 1) / mu. Drive a TURN-HEAVY calibration (arcs / spins), ~20-40 s.

Needs on the bag: /joint_states (measured wheel velocities) + a gyro (/ouster/imu or /imu/data).
Usage: python3 calibrate_turn_fit.py <bag_dir> [--mu 0.8] [--wheel-radius 0.35] [--half-track 0.365]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

GYRO_TOPICS = ("/ouster/imu", "/imu/data")
WHEEL_TOPIC = "/joint_states"
MAX_GYRO_DPS = 600.0  # drop single-sample glitch spikes (real motion is well under this)


def _hs(h) -> float:
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def _series(bag: Path, topic: str, extract):
    t, v = [], []
    with AnyReader([bag]) as r:
        for con, ts, raw in r.messages():
            if con.topic != topic:
                continue
            m = r.deserialize(raw, con.msgtype)
            t.append(_hs(m.header))
            v.append(extract(m))
    return np.asarray(t), np.asarray(v, float)


def fit(bag: Path, mu: float, R: float, B: float) -> dict:
    wt, wv = _series(bag, WHEEL_TOPIC, lambda m: (m.velocity[0], m.velocity[2]))  # (L, R)
    if len(wt) == 0:
        raise SystemExit(f"no {WHEEL_TOPIC} in {bag}")
    gyros = {}
    for tp in GYRO_TOPICS:
        gt, gv = _series(bag, tp, lambda m: (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z))
        if len(gt):
            gyros[tp] = (gt, gv)
    if not gyros:
        raise SystemExit(f"no gyro ({' / '.join(GYRO_TOPICS)}) in {bag}")

    # common 20 Hz clock over the overlap
    lo = max(wt[0], min(g[0][0] for g in gyros.values()))
    hi = min(wt[-1], max(g[0][-1] for g in gyros.values()))
    tg = np.arange(lo + 0.3, hi - 0.3, 0.05)
    smooth = lambda a, n=7: np.convolve(a, np.ones(n) / n, mode="same")
    wL = np.interp(tg, wt, wv[:, 0])
    wR = np.interp(tg, wt, wv[:, 1])
    combos = {"R-L": wR - wL, "R+L": wR + wL}

    # pick (gyro topic, axis, wheel combo) that best correlates -> yaw axis + turn convention
    best = None
    for tp, (gt, gv) in gyros.items():
        for ax in range(3):
            g = gv[:, ax].copy()
            g[np.abs(g) > np.deg2rad(MAX_GYRO_DPS)] = np.nan  # gate spikes
            gi = smooth(np.interp(tg, gt, np.nan_to_num(g)))
            for cname, cvec in combos.items():
                if gi.std() < 1e-6 or cvec.std() < 1e-6:
                    continue
                c = np.corrcoef(gi, cvec)[0, 1]
                if best is None or abs(c) > abs(best["corr"]):
                    best = dict(topic=tp, axis="xyz"[ax], corr=c, combo=cname, wz=gi, turn=cvec)
    wz, turn, combo = best["wz"], best["turn"], best["combo"]

    # robust fit over turning samples: yaw_rate = R*turn/(2B*alpha) -> alpha = R/(2B*slope)
    m = np.abs(turn) > 1.0
    n = int(m.sum())
    if n < 20:
        print(f"WARNING: only {n} turning samples -- drive more / sharper turns for a solid fit.")
    slope = np.sum(turn[m] * wz[m]) / np.sum(turn[m] ** 2)
    alpha = abs(R / (2 * B * slope))
    k_turn = (alpha - 1.0) / mu
    trap = getattr(np, "trapezoid", None) or np.trapz
    return dict(alpha=alpha, k_turn=k_turn, mu=mu, n=n, corr=best["corr"],
                gyro=f"{best['topic']} {best['axis']}", combo=combo,
                net_yaw_deg=float(np.degrees(trap(wz, tg))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag")
    ap.add_argument("--mu", type=float, default=0.8, help="planner's assumed friction")
    ap.add_argument("--wheel-radius", type=float, default=0.35)
    ap.add_argument("--half-track", type=float, default=0.365)
    a = ap.parse_args()
    r = fit(Path(a.bag), a.mu, a.wheel_radius, a.half_track)
    print(f"\n=== turn-gain fit: {a.bag} ===")
    print(f"  yaw truth : {r['gyro']}   turn term : {r['combo']}   (corr {r['corr']:+.2f}, n={r['n']})")
    print(f"  measured alpha = {r['alpha']:.2f}   (net yaw {r['net_yaw_deg']:+.0f} deg)")
    print(f"  ==> K_TURN = {r['k_turn']:.2f}   (at mu={r['mu']})")
    if abs(r["corr"]) < 0.7 or r["n"] < 20:
        print("  (!) low confidence -- record a longer, turn-heavier drive.")


if __name__ == "__main__":
    main()
