import numpy as np
import pytest
from pywiggle import utils, _wiggle
from fastgl import roots_legendre # pip install fastgl
import threadpoolctl as tp, sys

@pytest.mark.parametrize("lmax", [4, 8, 16, 32, 64, 1024, 2048])#, 4096, 8192, 16384])
def test_pywiggle_spin0(lmax):
    # This will compare wigner_d_00(ells) to numpy Legendre polynomials P_ells
    print(f"Testing test_pywiggle_spin0 with {lmax}")
    N = 2*lmax + 1

    info = tp.threadpool_info()
    openmps = [lib for lib in info if lib['internal_api'] == 'openmp']
    if (len(openmps) > 1) and sys.platform == "darwin":
        print("Loaded runtimes:", [lib['filepath'] for lib in openmps])
        sys.exit("❌  Multiple OpenMP runtimes loaded – This will seg-fault on Macs. This happens on Macs when healpy wheels are used, since it packages its own OpenMP which clashes with the homebrew OpenMP you needed to compile this package. Install healpy from source instead.")
    
    mu, w_mu = roots_legendre(N)
    P0 = np.polynomial.legendre.legvander(mu, lmax)
    P1 = utils.get_wigner_d(lmax,0,0,mu)
    bin_edges_trivial = np.arange(lmax+2)  # since bin_edges are of the form low_edge <= ell < upper_edge
    bin_weights_trivial = np.ones(lmax + 1)
    print("Series..")
    P2 = utils.get_wigner_d(lmax,0,0,mu,bin_edges_trivial,bin_weights_trivial)
    print("Double binned..")
    P2a,P2b = utils.get_wigner_d(lmax,0,0,mu,bin_edges_trivial,bin_weights_trivial,bin_weights_trivial)
    print("Legendre..")
    Pl = _wiggle._compute_legendre_matrix(lmax,mu)
    assert np.allclose(P0,Pl, atol=1e-14)
    assert np.allclose(P0,P1, atol=1e-14)
    assert np.allclose(P0,P2, atol=1e-14)
    assert np.allclose(P0,P2a, atol=1e-14)
    assert np.allclose(P0,P2b, atol=1e-14)


@pytest.mark.parametrize("lmax", [4, 8, 16, 32, 64, 1000, 2000, 4000])
def test_pywiggle_spin2(lmax):
    # This will compare wigner_d_20(ells) and wigner_d_22(ells) to the implementation in the wigner PyPI package
    # This begins to disagree at the 1e-14 level at ell>64

    N = 2*lmax + 1
    mu, w_mu = roots_legendre(N)
    bin_edges_trivial = np.arange(lmax+2)  # since bin_edges are of the form low_edge <= ell < upper_edge
    bin_weights_trivial = np.ones(lmax + 1)
    
    m1=2
    for m2 in [2,0]:
        print(f"Testing test_pywiggle_spin2 with lmax={lmax},m1={m1},m2={m2}")
        P0 = utils.get_wigner_d(lmax,m1,m2,mu)
        P1 = utils.get_wigner_d(lmax,m1,m2,mu,bin_edges_trivial,bin_weights_trivial)
        P2a,P2b = utils.get_wigner_d(lmax,m1,m2,mu,bin_edges_trivial,bin_weights_trivial,bin_weights_trivial)
        assert np.allclose(P0,P1, atol=1e-14)
        assert np.allclose(P0,P2a, atol=1e-14)
        assert np.allclose(P0,P2b, atol=1e-14)


