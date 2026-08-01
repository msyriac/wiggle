"""Time pywiggle coupling-matrix computations for branch comparison.

Runs a fixed set of mode-coupling-matrix scenarios and prints per-scenario
timings. Select which code version to benchmark with --package-path, which
bypasses the editable install by pointing at an alternate directory that
contains a ``pywiggle`` package (e.g. a ``git archive`` of the main branch
with the compiled extension copied in). Without it, the currently installed
(working-tree) pywiggle is used.

Example
-------
    python scripts/bench_branches.py --label optimizations
    python scripts/bench_branches.py --label main --package-path /tmp/wiggle_main
"""

import argparse
import sys
import time

import numpy as np


def load_pywiggle(package_path):
    """Import and return the pywiggle module, optionally from an explicit path.

    Parameters
    ----------
    package_path : str or None
        Directory containing a ``pywiggle`` package to import instead of the
        installed one. When given, the meson-python editable-install meta-path
        finder (which would otherwise shadow ``sys.path``) is removed first.

    Returns
    -------
    module
        The imported ``pywiggle`` module.
    """
    if package_path is not None:
        sys.meta_path = [
            f for f in sys.meta_path if type(f).__name__ != "MesonpyMetaFinder"
        ]
        sys.path.insert(0, package_path)
    import pywiggle

    print(f"pywiggle loaded from: {pywiggle.__file__}")
    return pywiggle


def timeit(f, repeats):
    """Run callable ``f`` ``repeats`` times and return (best, all_times) in seconds."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts), ts


def main():
    """Run the branch-comparison timing scenarios and print results."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="current", help="Label for this code version")
    p.add_argument("--package-path", default=None, help="Alternate pywiggle location")
    p.add_argument("--lmax", type=int, default=2048)
    p.add_argument("--nmask", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--debug", action="store_true", help="Quick test at low lmax")
    args = p.parse_args()
    if args.debug:
        args.lmax = 512
        args.repeats = 2

    pywiggle = load_pywiggle(args.package_path)
    lmax = args.lmax
    rng = np.random.default_rng(0)
    mcls1 = np.abs(rng.standard_normal(2 * lmax + 1)) / np.arange(1, 2 * lmax + 2) ** 2
    mclsN = (
        np.abs(rng.standard_normal((args.nmask, 2 * lmax + 1)))
        / np.arange(1, 2 * lmax + 2) ** 2
    )

    results = {}

    # Scenario 1: object construction (GL nodes + spin-0 Wigner-d build)
    t, ts = timeit(lambda: pywiggle.Wiggle(lmax), args.repeats)
    results["init"] = t
    print(f"[{args.label}] init: best {t:.3f} s  all {[f'{x:.3f}' for x in ts]}")

    # Persistent object for the matrix scenarios (init cost excluded)
    w = pywiggle.Wiggle(lmax)

    def run(name, f):
        t, ts = timeit(f, args.repeats)
        results[name] = t
        print(f"[{args.label}] {name}: best {t:.3f} s  all {[f'{x:.3f}' for x in ts]}")

    run("TT_1mask", lambda: w.get_coupling_matrix_from_mask_cls(mcls1, "TT"))
    run(
        f"TT_{args.nmask}mask", lambda: w.get_coupling_matrix_from_mask_cls(mclsN, "TT")
    )

    # Spin-2 pair on a fresh object each call: captures both the (2,2)
    # Wigner-d build (cached after first use) and any +/- work sharing.
    def spin2_pair():
        wf = pywiggle.Wiggle(lmax)
        wf.get_coupling_matrix_from_mask_cls(mcls1, "+")
        wf.get_coupling_matrix_from_mask_cls(mcls1, "-")

    run("EEBB_pair_cold", spin2_pair)

    # Warm spin-2 pair: Wigner-d tables already built on w
    def spin2_pair_warm():
        w.get_coupling_matrix_from_mask_cls(mcls1, "+")
        w.get_coupling_matrix_from_mask_cls(mcls1, "-")

    run("EEBB_pair_warm", spin2_pair_warm)

    run("TE_1mask", lambda: w.get_coupling_matrix_from_mask_cls(mcls1, "TE"))

    print("RESULT", args.label, {k: round(v, 4) for k, v in results.items()})


if __name__ == "__main__":
    main()
