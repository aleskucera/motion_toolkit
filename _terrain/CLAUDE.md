# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Code Style (this project)

`heightmap/` and `traversability/` are the canonical examples of everything below — mirror them.

- **Type hints everywhere.** Annotate every function/method signature — parameters *and* return — including private helpers. Exception: `@wp.kernel` functions carry Warp's own array/scalar annotations; don't add Python hints there. `from __future__ import annotations` is on in every code module (the package `__init__.py` re-export files are the only exception), so use modern forms (`tuple[float, float]`, `X | None`, `wp.array`).
- **Formatters: black (line-length 100) + reorder-python-imports** (black pinned in `pyproject.toml [tool.black]`). Imports are one symbol per line, sorted — `from x import a` / `from x import b`, never `from x import a, b`. Lint with ruff. Run the formatters before committing; don't hand-format against them.
- **Names carry the meaning.** Prefer a descriptive name that needs no comment to decode (`resolution`, `smooth_sigma`, `support_ratio`) over a short name plus an explanation. Keep terse names only for grid/loop indices and math (`i, j` = row/col, `p`, `dx, dy`).
- **Comments explain WHY, not what.** Match the existing density: short notes on rationale, units, and grid/frame conventions — never a restatement of the code. Keep units and frame conventions explicit (e.g. `# row=y, col=x; sigma in CELLS not meters`). A trailing comment that would push its line past 100 goes on its own line *above* the code (otherwise black wraps the code around it).

## 6. Device-Native by Default (this project)

This is a Warp/CUDA codebase. Data-parallel work runs on-device; **host↔device copies are the dominant performance cost, and a per-frame upload→compute→readback round trip is the worst thing you can write here.** Default to device-resident `wp.array`, NOT numpy. Do not treat numpy as the baseline and device as an optimization — it is the reverse.

- **Bulk data → device.** Point clouds, grids, per-point / per-cell fields, and *any new data-parallel stage* are `wp.array` + `@wp.kernel`. Never implement a compute stage in numpy, and never introduce a host↔device round trip for bulk data.
- **numpy only at unavoidable boundaries:** parsing an incoming ROS / message payload, and small host-side *control* values — 4×4 poses, scalars, gate thresholds, counts. A pose stays numpy (16 floats driving host control flow); a cloud never does.
- **If a stage seems to need numpy, STOP and say so before writing it.** Surface the round-trip cost and propose the device-native path — do not silently fall back to host arrays. The user has not, and does not, approve numpy for bulk data by default.
- Mirror `icp/`, `voxel.py`, and `mapping/` (esp. `DeviceMapAccumulator`): device-resident data + kernels are the norm here, not the exception.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
