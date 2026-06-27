"""Per-stage GPU timing for CUDA-graph-captured pipelines: captured CUDA events + online mean/std.

Drop boundary marks (`mark(k)`) into a graph-captured method; placed inside the capture they become
graph nodes that time each replay. After a replay, `accumulate()` reads the intervals and folds them
into a running mean/std (Welford). CUDA-only and opt-in: when disabled it records nothing, so the
captured graph and the default path are byte-for-byte unchanged.

The catch: `accumulate()` reads events via `get_event_elapsed_time`, which SYNCS on the end event --
a profiling run is therefore serialized. Use the per-stage means as the cost, NOT the run's wall-clock.
"""

from __future__ import annotations

import warp as wp


class StageProfiler:
    """N named stages -> N+1 boundary events. `mark(0..N)` at the boundaries, `accumulate()` after a
    replay, `stats()` for the running per-stage mean/std/n. The caller skips the first (build/warmup)
    sample. A no-op when `enabled` is False or the device isn't CUDA."""

    def __init__(self, device, stage_names, enabled):
        self.device = wp.get_device(device)
        self.stage_names = tuple(stage_names)
        self.enabled = bool(enabled) and self.device.is_cuda
        self._ev = (
            [
                wp.Event(device=self.device, enable_timing=True)
                for _ in range(len(self.stage_names) + 1)
            ]
            if self.enabled
            else None
        )
        self.reset()

    def reset(self):
        """Clear the accumulated stats (e.g. after a warmup run)."""
        self._stats = {name: [0, 0.0, 0.0] for name in self.stage_names}  # [n, mean, M2] (Welford)

    def mark(self, k):
        """Record boundary event k (a graph node when inside a capture). No-op when disabled."""
        if self.enabled:
            wp.record_event(self._ev[k])

    def accumulate(self):
        """Read the per-stage intervals off the last replay and fold into the running mean/std."""
        if not self.enabled:
            return
        ev = self._ev
        for i, name in enumerate(self.stage_names):
            x = wp.get_event_elapsed_time(ev[i], ev[i + 1])  # syncs on ev[i+1]
            acc = self._stats[name]
            acc[0] += 1
            delta = x - acc[1]
            acc[1] += delta / float(acc[0])
            acc[2] += delta * (x - acc[1])

    def stats(self):
        """{stage: {"mean_ms", "std_ms", "n"}} over all accumulated replays."""
        out = {}
        for name in self.stage_names:
            n, mean, m2 = self._stats[name]
            std = (m2 / float(n - 1)) ** 0.5 if n > 1 else 0.0
            out[name] = {"mean_ms": mean, "std_ms": std, "n": n}
        return out
