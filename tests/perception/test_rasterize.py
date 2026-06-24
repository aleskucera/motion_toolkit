"""Guard the point-cloud -> heightmap rasterizer's frame convention: an asymmetric round-trip plus a
transpose/handedness check. This is the sim-to-real seam; a row<->col swap or an x/y flip here would
silently mis-place every obstacle, so these tests are the tripwire.

  python -m tests.perception.test_rasterize
"""
import numpy as np

from kinematic_helhest.perception.rasterize import heightmap_to_points
from kinematic_helhest.perception.rasterize import rasterize


def selftest_transpose_guard():
    # ny != nx so a transpose also changes shape; two points whose (row, col) are SWAPS of each other
    # with DISTINCT heights -> if the code swapped r<->c, A and B would trade cells (and heights).
    x0, y0, cell, ny, nx = 0.0, 0.0, 0.1, 20, 30
    pts = np.array([[1.2, 0.3, 1.5],   # x=1.2 -> col 12, y=0.3 -> row 3
                    [0.3, 1.2, 0.4]])  # x=0.3 -> col 3,  y=1.2 -> row 12
    H, known = rasterize(pts, x0, y0, cell, ny, nx)
    okA = bool(known[3, 12]) and abs(H[3, 12] - 1.5) < 1e-5  # A at row 3, col 12
    okB = bool(known[12, 3]) and abs(H[12, 3] - 0.4) < 1e-5  # B at row 12, col 3
    print(f"  A@[3,12]={H[3,12]:.2f} (want 1.5)   B@[12,3]={H[12,3]:.2f} (want 0.4)")
    print(f"transpose/handedness guard  {'OK' if okA and okB else 'FAIL'}")
    return okA and okB


def selftest_roundtrip():
    # an ASYMMETRIC heightmap (depends differently on row vs col + an off-diagonal spike) -> a transpose
    # or flip anywhere in rasterize/heightmap_to_points breaks the recovery.
    x0, y0, cell, ny, nx = -1.0, 2.0, 0.05, 24, 40
    rr, cc = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    H0 = (0.03 * rr + 0.07 * cc).astype(np.float32)
    H0[5, 30] = 2.0
    H1, known = rasterize(heightmap_to_points(H0, x0, y0, cell), x0, y0, cell, ny, nx)
    err = float(np.abs(H1 - H0).max())
    print(f"  heightmap round-trip  max|dH|={err:.2e}  known_all={bool(known.all())}")
    print(f"round-trip  {'OK' if err < 1e-5 and known.all() else 'FAIL'}")
    return err < 1e-5 and known.all()


if __name__ == "__main__":
    a = selftest_transpose_guard()
    b = selftest_roundtrip()
    print("ALL OK" if a and b else "FAILED")
