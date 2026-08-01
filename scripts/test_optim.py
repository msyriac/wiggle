"""Prototype and benchmark candidate optimizations for wiggle's multi-mask
unbinned TT mode-coupling matrix computation.

Variants tested against the pywiggle baseline (all exploit the identity
M[m] = b1^T diag(W_m) b2 with b1 = P, b2 = ((2l+1)/2) P):

  gemm    : batch all masks into one large BLAS GEMM instead of einsum
  parity  : exploit mu -> -mu symmetry of GL nodes to halve GEMM flops
  psyrk   : parity + DSYRK for the symmetric ee/oo blocks (~4x fewer flops)
  f32     : float32 version of psyrk (~8x fewer effective flops)

Also times ducc0's coupling_matrix_rect on the same inputs for reference.
All variants are checked for accuracy against the pywiggle baseline.

Usage:
  python scripts/test_optim.py [--lmax 2048] [--nmasks 1 4 10] [--nthreads 12]
                               [--repeats 2] [--debug]
"""

import argparse
import time

import numpy as np
from scipy.linalg import blas as sblas

import ducc0
import pywiggle


def _timeit(fn, repeats):
    """Run fn() repeats times, return (best_time_seconds, last_result)."""
    best = np.inf
    res = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t0)
    return best, res


def _get_W(w, mcls):
    """GL-weighted mask correlation functions W (N, nmask) from mask cls (nmask, 2lmax+1)."""
    coeff = (2 * w.ells + 1) / (4 * np.pi) * mcls  # (m, nl)
    xi = w.cd00 @ coeff.T  # (N, m)
    return w.w_mu[:, None] * xi


def mcm_baseline(w, mcls):
    """Installed pywiggle multi-mask TT coupling matrix (now parity-halved)."""
    return w.get_coupling_matrix_from_mask_cls(mcls, "TT")


def mcm_einsum_full(w, mcls):
    """Old full-grid einsum path (pre-parity-halving pywiggle).

    Replicates the original unbinned contraction M[m] = b1^T diag(W_m) b2 on
    all N = 2lmax+1 GL nodes with the same two einsums the old core used.
    Serves as the accuracy reference and the 'baseline' timing row.
    """
    L = w.lmax + 1
    W = _get_W(w, mcls).T  # (m, N)
    b1 = w.ud00  # (N, L)
    b2 = ((2 * np.arange(L) + 1) / 2.0) * w.ud00  # (N, L)
    D = np.einsum("mi,ik->mik", W, b2, optimize="greedy")
    return np.einsum("ij,mik->mjk", b1, D, optimize="greedy")


def mcm_gemm(w, mcls, b2=None):
    """Batch all masks into a single (L, N) x (N, m*L) BLAS GEMM.

    E[i, m, k] = W[i, m] * b2[i, k] is built contiguously in (N, m, L) order so
    the reshape to (N, m*L) is free; one GEMM then contracts over the GL nodes.
    """
    L = w.lmax + 1
    W = _get_W(w, mcls)  # (N, m)
    if b2 is None:
        b2 = ((2 * np.arange(L) + 1) / 2.0) * w.ud00  # (N, L)
    m = W.shape[1]
    E = W[:, :, None] * b2[:, None, :]  # (N, m, L)
    M = w.ud00.T @ E.reshape(W.shape[0], m * L)  # (L, m*L)
    return M.reshape(L, m, L).transpose(1, 0, 2)


def _parity_setup(w):
    """Precompute half-node column-parity-split b1/b2 factors for the parity trick.

    Uses P_l(-mu) = (-1)^l P_l(mu) and the symmetry of GL nodes: contributions
    from node pairs (mu, -mu) combine into W+ = W(mu)+W(-mu) for (j+k) even and
    W- = W(mu)-W(-mu) for (j+k) odd. Splitting columns by parity turns this
    into four half-size products. Returns dict of split matrices.
    """
    L = w.lmax + 1
    N = len(w.mu)
    assert np.allclose(w.mu, -w.mu[::-1]), "GL nodes not symmetric"
    h = N // 2
    nh = h + (N % 2)  # include middle node row if N odd
    nw2 = (2 * np.arange(L) + 1) / 2.0
    b1h = w.ud00[:nh]  # (nh, L)
    b2h = nw2 * b1h
    ps = dict(
        h=h,
        nh=nh,
        L=L,
        b1e=np.ascontiguousarray(b1h[:, ::2]),
        b1o=np.ascontiguousarray(b1h[:, 1::2]),
        b2e=np.ascontiguousarray(b2h[:, ::2]),
        b2o=np.ascontiguousarray(b2h[:, 1::2]),
        nw2e=nw2[::2],
        nw2o=nw2[1::2],
    )
    for k in ("b1e", "b1o", "b2e", "b2o"):
        ps[k + "32"] = ps[k].astype(np.float32)
    return ps


def _parity_W(w, mcls, ps):
    """Half-node symmetric/antisymmetric GL-weighted correlation functions."""
    W = _get_W(w, mcls)  # (N, m)
    h, nh = ps["h"], ps["nh"]
    Wrev = W[::-1]
    Wp = np.empty((nh, W.shape[1]))
    Wm = np.empty((nh, W.shape[1]))
    Wp[:h] = W[:h] + Wrev[:h]
    Wm[:h] = W[:h] - Wrev[:h]
    if nh > h:  # middle node mu=0: count once, odd-l rows vanish there
        Wp[h] = W[h]
        Wm[h] = 0.0
    return Wp, Wm


def mcm_parity(w, mcls, ps, dtype=np.float64):
    """Parity-halved variant: four half-size batched GEMMs (~2x fewer flops).

    Set dtype=np.float32 to run the GEMMs in single precision (sgemm), which
    roughly doubles BLAS throughput at the cost of ~1e-7 absolute accuracy
    (relative to the matrix maximum).
    """
    L, nh = ps["L"], ps["nh"]
    m = mcls.shape[0]
    Wp, Wm = _parity_W(w, mcls, ps)
    if dtype != np.float64:
        Wp = Wp.astype(dtype)
        Wm = Wm.astype(dtype)
    M = np.empty((m, L, L), dtype=dtype)

    sfx = "" if dtype == np.float64 else "32"

    def blk(b1, b2, Wblk):
        E = Wblk[:, :, None] * b2[:, None, :]  # (nh, m, Lk)
        R = b1.T @ E.reshape(nh, -1)  # (Lj, m*Lk)
        return R.reshape(b1.shape[1], m, b2.shape[1]).transpose(1, 0, 2)

    M[:, ::2, ::2] = blk(ps["b1e" + sfx], ps["b2e" + sfx], Wp)
    M[:, 1::2, 1::2] = blk(ps["b1o" + sfx], ps["b2o" + sfx], Wp)
    M[:, ::2, 1::2] = blk(ps["b1e" + sfx], ps["b2o" + sfx], Wm)
    M[:, 1::2, ::2] = blk(ps["b1o" + sfx], ps["b2e" + sfx], Wm)
    return M


def mcm_psyrk(w, mcls, ps, dtype=np.float64):
    """Parity + SYRK variant (~4x fewer flops than baseline).

    The ee and oo blocks are S = A^T A with A = sqrt(W+) * P (W+ >= 0 for
    non-negative masks), computed triangle-only with {d,s}syrk and scaled by
    (2k+1)/2 on the columns afterwards. The eo block is one half-size GEMM and
    gives oe by transposition/rescaling. Falls back to GEMM if W+ has negative
    entries. Set dtype=np.float32 for the f32 variant.
    """
    L = ps["L"]
    m = mcls.shape[0]
    Wp, Wm = _parity_W(w, mcls, ps)
    syrk = sblas.dsyrk if dtype == np.float64 else sblas.ssyrk
    b1e = ps["b1e"].astype(dtype, copy=False)
    b1o = ps["b1o"].astype(dtype, copy=False)
    M = np.empty((m, L, L), dtype=dtype)

    # For a non-negative mask, xi >= 0 so Wp >= 0 up to roundoff; clip tiny
    # negatives, but refuse genuinely negative correlation functions.
    tol = -1e-13 * np.max(Wp)
    if np.any(Wp < tol):  # cannot take sqrt (non-physical mask spectrum)
        raise ValueError("W+ has negative entries; SYRK variant not applicable")
    sqWp = np.sqrt(np.maximum(Wp, 0.0)).astype(dtype)  # (nh, m)

    for i in range(m):
        Ae = sqWp[:, i : i + 1] * b1e  # (nh, Le)
        Ao = sqWp[:, i : i + 1] * b1o
        See = syrk(1.0, Ae, trans=1)  # upper triangle of Ae^T Ae
        Soo = syrk(1.0, Ao, trans=1)
        See = See + np.triu(See, 1).T  # symmetrize
        Soo = Soo + np.triu(Soo, 1).T
        Seo = b1e.T @ ((Wm[:, i : i + 1] * b1o).astype(dtype, copy=False))
        M[i, ::2, ::2] = See * ps["nw2e"].astype(dtype)
        M[i, 1::2, 1::2] = Soo * ps["nw2o"].astype(dtype)
        M[i, ::2, 1::2] = Seo * ps["nw2o"].astype(dtype)
        M[i, 1::2, ::2] = Seo.T * ps["nw2e"].astype(dtype)
    return M


def mcm_ducc(mcls, lmax, nthreads, dtype=np.float64):
    """ducc0 coupling_matrix_rect on nmask spectra; output scaled to MASTER convention."""
    m = mcls.shape[0]
    res = np.empty((m, lmax + 1, lmax + 1), dtype=dtype)
    ducc0.misc.experimental.coupling_matrix_rect(
        mcls, tuple([0] * m), res=res, nthreads=nthreads
    )
    return res * (2 * np.arange(lmax + 1) + 1.0)[None, None, :]


def mcm_ducc_new(mcls, lmax, nthreads, dtype=np.float64):
    """ducc0 coupling_matrix_rect_new (fcomp-table algorithm, ducc >= 0.42.0 ChangeLog).

    The input spectra must stay float64; only the output dtype is templated
    (accumulation remains f64), selected by the res array's dtype.
    """
    m = mcls.shape[0]
    res = np.empty((m, lmax + 1, lmax + 1), dtype=dtype)
    ducc0.misc.experimental.coupling_matrix_rect_new(
        mcls, tuple([0] * m), res=res, nthreads=nthreads
    )
    return res * (2 * np.arange(lmax + 1) + 1.0)[None, None, :]


def make_mask_cls(w, nmask, rng):
    """Pseudo-spectra of non-negative axisymmetric apodized-cap masks.

    An axisymmetric mask m(mu) >= 0 has a_l0 = 2*pi*sqrt((2l+1)/(4pi)) *
    int m P_l dmu and C_l = a_l0^2/(2l+1); the integral is exact on the GL
    grid. This guarantees a physical (non-negative-map) mask spectrum, for
    which the mask correlation function xi(mu) is non-negative.
    """
    nl = w.cd00.shape[1]
    mcls = np.empty((nmask, nl))
    for i in range(nmask):
        mu0 = rng.uniform(-0.2, 0.5)  # cap edge
        sig = rng.uniform(0.005, 0.02)  # apodization width
        m = 0.5 * (1 + np.tanh((w.mu - mu0) / sig))  # smooth step in mu
        # add non-negative small-scale structure (point-source-like holes)
        for _ in range(20):
            muh = rng.uniform(-1, 1)
            m = m * (1 - np.exp(-((w.mu - muh) ** 2) / (2 * 0.001**2)))
        integ = (w.w_mu * m) @ w.cd00  # (nl,) = int m P_l dmu
        # a_l0 = 2*pi*sqrt((2l+1)/(4pi))*integ ; C_l = a_l0^2/(2l+1) = pi*integ^2
        mcls[i] = np.pi * integ**2
    return mcls


def relerr(A, B):
    """Accuracy metrics between A and reference B.

    Returns (max relative error over entries above 1e-6 of the max magnitude,
    max absolute error normalized by the max magnitude). The two together
    characterize accuracy for matrices with large dynamic range.
    """
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    bmax = np.max(np.abs(B))
    sel = np.abs(B) > 1e-6 * bmax
    r = np.max(np.abs(A[sel] - B[sel]) / np.abs(B[sel]))
    a = np.max(np.abs(A - B)) / bmax
    return r, a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmax", type=int, default=2048)
    p.add_argument("--nmasks", type=int, nargs="+", default=[1, 4, 10])
    p.add_argument("--nthreads", type=int, default=12)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--debug", action="store_true", help="quick test at low lmax")
    args = p.parse_args()
    if args.debug:
        args.lmax = 256
        args.nmasks = [1, 3]
    lmax = args.lmax
    print(
        f"lmax={lmax}  nmasks={args.nmasks}  nthreads={args.nthreads} "
        f"repeats={args.repeats}"
    )

    rng = np.random.default_rng(1)

    t0 = time.perf_counter()
    w = pywiggle.Wiggle(lmax, verbose=False)
    print(
        f"[setup] Wiggle precompute: {time.perf_counter() - t0:.3f}s "
        f"(N={len(w.mu)} GL nodes)"
    )
    t0 = time.perf_counter()
    ps = _parity_setup(w)
    print(f"[setup] parity split precompute: {time.perf_counter() - t0:.3f}s")
    nb = 50
    bin_edges = np.linspace(2, lmax, nb + 1).astype(int)
    t0 = time.perf_counter()
    wb = pywiggle.Wiggle(lmax, bin_edges=bin_edges, verbose=False)
    print(f"[setup] binned Wiggle precompute: {time.perf_counter() - t0:.3f}s")

    for nmask in args.nmasks:
        # physical (non-negative map) mask pseudo-spectra with variation
        mcls = make_mask_cls(w, nmask, rng)
        print(f"\n=== nmask={nmask} ===")

        t_base, Mb = _timeit(lambda: mcm_einsum_full(w, mcls), args.repeats)
        if Mb.ndim == 2:
            Mb = Mb[None]
        print(f"baseline (einsum) : {t_base:8.3f}s")

        for name, fn in [
            ("pywiggle-installed", lambda: mcm_baseline(w, mcls)),
            ("gemm", lambda: mcm_gemm(w, mcls)),
            ("parity", lambda: mcm_parity(w, mcls, ps)),
            ("parity-f32", lambda: mcm_parity(w, mcls, ps, dtype=np.float32)),
            ("psyrk", lambda: mcm_psyrk(w, mcls, ps)),
        ]:
            t, M = _timeit(fn, args.repeats)
            r, a = relerr(M, Mb)
            print(
                f"{name:18s}: {t:8.3f}s  speedup vs baseline: "
                f"{t_base / t:5.2f}x  relerr: {r:.2e}  abserr/max: {a:.2e}"
            )

        for name, fn in [
            ("ducc-rectnew-f64", lambda: mcm_ducc_new(mcls, lmax, args.nthreads)),
            (
                "ducc-rectnew-f32o",
                lambda: mcm_ducc_new(mcls, lmax, args.nthreads, np.float32),
            ),
            ("ducc-rect-f64", lambda: mcm_ducc(mcls, lmax, args.nthreads)),
        ]:
            t, M = _timeit(fn, args.repeats)
            r, a = relerr(M, Mb)
            print(
                f"{name:18s}: {t:8.3f}s  speedup vs baseline: "
                f"{t_base / t:5.2f}x  relerr: {r:.2e}  abserr/max: {a:.2e}"
            )

        # binned multi-mask path: wiggle computes binned matrices directly
        # (no lmax^2 matrix is ever formed); no ducc equivalent exists.
        t, Mbin = _timeit(
            lambda: wb.get_coupling_matrix_from_mask_cls(mcls, "TT"), args.repeats
        )
        print(
            f"binned ({nb} bins) : {t:8.3f}s  speedup vs baseline: "
            f"{t_base / t:5.2f}x  (different output: {Mbin.shape})"
        )


if __name__ == "__main__":
    main()
