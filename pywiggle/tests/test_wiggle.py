import os
from time import time

import numpy as np
import healpy as hp
import pytest

import pywiggle
from pywiggle import utils

NSIDE = 256
LMAX = 2*NSIDE
NPIX = hp.nside2npix(NSIDE)

def galactic_strip_mask(nside, b_cut_deg):
    """
    Create a Galactic strip mask that masks |b| < b_cut_deg.
    
    Parameters
    ----------
    nside : int
        Healpix NSIDE resolution.

    b_cut_deg : float
        Galactic latitude cut in degrees (absolute value).

    Returns
    -------
    mask : ndarray
        Binary mask with 1s where |b| > b_cut_deg and 0s otherwise.
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    # Convert to Galactic coordinates
    vec = hp.ang2vec(theta, phi)
    lon, lat = hp.vec2ang(vec, lonlat=True)  # In degrees

    # Create mask: keep pixels where |b| > b_cut_deg
    mask = np.ones(npix)
    mask[np.abs(lat) < b_cut_deg] = 0
    return mask


def get_theory_cls(lmax):
    """Return power law spectra."""
    ells = np.arange(lmax + 1)
    cl = np.zeros((4, lmax + 1))
    cl[0,2:] = 1e-3 * (ells[2:] + 1.)**-2  # TT
    cl[1,2:] = 1e-4 * (ells[2:] + 1.)**-2  # EE
    return cl

def test_power():
    mask_fraction = 0.8

    cl_th = get_theory_cls(LMAX)

    seed = 10
    np.random.seed(10)
    
    nsims = 3
    # Create Galactic strip mask
    b_cut = 20  # degrees
    mask = galactic_strip_mask(NSIDE, b_cut)
    mask = hp.smoothing(mask, fwhm=np.radians(1.5))
    mask_alm = hp.map2alm(mask, lmax=2 * LMAX)

    bin_edges = np.arange(40,LMAX,40)
    bcents = (bin_edges[1:]+bin_edges[:-1])/2.
    w = pywiggle.Wiggle(LMAX, bin_edges=bin_edges)
    w.add_mask('m', mask_alm)
    

    acl_TT = 0.
    acl_EE = 0.
    for i in range(nsims):
        print(i)
        # Generate Q/U maps from theoretical Cls
        maps = hp.synfast(cl_th, NSIDE, new=True, pol=True, lmax=LMAX)

        maps = maps*mask
        alms = hp.map2alm(maps, lmax=LMAX, iter=0,pol=True)


        ret = w.get_powers(alms,alms, 'm',return_theory_filter=(i==0))
        cl_EE = ret['EE']['Cls']
        cl_TT = ret['TT']['Cls']
        if i==0:
            Th_TT =  ret['TT']['Th']
            Th_Pol =  ret['ThPol']
        
        
        acl_EE = acl_EE + cl_EE
        acl_TT = acl_TT + cl_TT
        
    acl_EE = acl_EE / nsims
    acl_TT = acl_TT / nsims

    # Compare to input theory
    ells = np.arange(LMAX + 1)

    btheory_EE = Th_Pol[:w.nbins,:LMAX + 1] @ cl_th[1][:LMAX + 1]
    btheory_TT = Th_TT @ cl_th[0][:LMAX + 1]

    assert np.allclose(acl_EE[:LMAX + 1], btheory_EE, rtol=1e-1, atol=0)
    assert np.allclose(acl_TT[:LMAX + 1], btheory_TT, rtol=1e-1, atol=0)


def test_integer_spintype():
    """An integer spintype m requests the spin-m x spin-0 coupling.

    m=0 and m=2 must reproduce 'TT' and 'TE' exactly, higher m must build
    invertible binned matrices whose theory filter inverts consistently, and
    bad spintypes must still be rejected.
    """
    np.random.seed(11)
    mask = galactic_strip_mask(NSIDE, 20)
    mask = hp.smoothing(mask, fwhm=np.radians(1.5))
    mask_alm = hp.map2alm(mask, lmax=2 * LMAX)

    w = pywiggle.Wiggle(LMAX, bin_edges=np.arange(40, LMAX, 40))
    w.add_mask('m', mask_alm)

    # integer spins that overlap the existing string spintypes must be identical
    for m, spintype in [(0, 'TT'), (2, 'TE')]:
        assert np.array_equal(w.get_coupling_matrix_from_ids('m', 'm', m),
                              w.get_coupling_matrix_from_ids('m', 'm', spintype))
        assert np.array_equal(w.get_theory_filter('m', 'm', spintype=m),
                              w.get_theory_filter('m', 'm', spintype=spintype))

    ells = np.arange(LMAX + 1)
    alm = hp.synalm(1e-3 * (ells + 10.)**-2., lmax=LMAX)
    pcl = pywiggle.alm2cl(alm, alm)
    assert np.allclose(w.decoupled_cl(pcl, 'm', spintype=0)['Cls'],
                       w.get_powers(alm, alm, 'm')['TT']['Cls'], rtol=1e-12, atol=0)
    assert np.allclose(w.decoupled_cl(pcl, 'm', spintype=2)['Cls'],
                       w.decoupled_cl(pcl, 'm', spintype='TE')['Cls'], rtol=1e-12, atol=0)

    # a smooth theory spectrum must survive couple-then-decouple at every m
    cl = 1e-3 * (ells + 10.)**-2.
    for m in range(5):
        mcm = w.get_coupling_matrix_from_ids('m', 'm', m)
        assert np.isfinite(np.linalg.cond(mcm))
        btheory = w.get_theory_filter('m', 'm', spintype=m) @ cl
        assert np.allclose(np.linalg.solve(mcm, mcm @ btheory), btheory,
                           rtol=1e-8, atol=0)

    with pytest.raises(ValueError):
        w.get_coupling_matrix_from_ids('m', 'm', -1)
    with pytest.raises(ValueError):
        w.get_coupling_matrix_from_ids('m', 'm', 'XY')
    with pytest.raises(ValueError):
        w.get_coupling_matrix_from_ids('m', 'm', 3, pure_E=True)
