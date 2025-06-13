import os
from time import time
try:
    nthreads = int(os.environ["OMP_NUM_THREADS"])
except:
    import multiprocessing
    nthreads = multiprocessing.cpu_count()

import numpy as np
import healpy as hp
import ducc0
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
    """Return some fake theory spectra (e.g. flat or decaying power law)."""
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

    acl_TT = 0.
    acl_EE = 0.
    for i in range(nsims):
        print(i)
        # Generate Q/U maps from theoretical Cls
        maps = hp.synfast(cl_th, NSIDE, new=True, pol=True, lmax=LMAX)



        maps = maps*mask
        alms = hp.map2alm(maps, lmax=LMAX, iter=0,pol=True)
        alm_t = alms[0]
        alm_e = alms[1]
        alm_b = alms[2]

        bin_edges = np.arange(40,LMAX,40)
        bcents = (bin_edges[1:]+bin_edges[:-1])/2.

        # Estimate decoupled Cls
        cl_EE, tf_EE, cl_BB, tf_BB = pywiggle.alm2auto_power_spin2(
            LMAX, alm_e, alm_b, mask_alm,
            bin_edges=bin_edges,
            return_theory_filter=True
        )
        
        # Estimate decoupled Cls
        cl_TT, tf_TT = pywiggle.alm2auto_power_spin0(
            LMAX, alm_t, mask_alm,
            bin_edges=bin_edges,
            return_theory_filter=True
        )
        
        acl_EE = acl_EE + cl_EE
        acl_TT = acl_TT + cl_TT
        
    acl_EE = acl_EE / nsims
    acl_TT = acl_TT / nsims

    # Compare to input theory
    ells = np.arange(LMAX + 1)

    btheory_EE = tf_EE @ cl_th[1][:LMAX + 1]
    btheory_TT = tf_TT @ cl_th[0][:LMAX + 1]
    
    assert np.allclose(acl_EE[:LMAX + 1], btheory_EE, rtol=0.1, atol=1e-6)
    assert np.allclose(acl_TT[:LMAX + 1], btheory_TT, rtol=0.1, atol=1e-6)


# Copied these from mreinecke/ducc0
def tri2full(tri, lmax):
    res = np.zeros((tri.shape[0], tri.shape[1], lmax+1, lmax+1))
    lfac = 2.*np.arange(lmax+1) + 1.
    for l1 in range(lmax+1):
        startidx = l1*(lmax+1) - (l1*(l1+1))//2
        res[:,:,l1,l1:] = lfac[l1:] * tri[:,:, startidx+l1:startidx+lmax+1]
        res[:,:,l1:,l1] = (2*l1+1) * tri[:,:, startidx+l1:startidx+lmax+1]
    return res
    
def mcm00_ducc_tri(spec, lmax):
    out= np.empty((spec.shape[0],1,((lmax+1)*(lmax+2))//2),dtype=np.float32)
    ducc0.misc.experimental.coupling_matrix_spin0and2_tri(spec.reshape((spec.shape[0],1,spec.shape[1])), lmax, (0,0,0,0), (0,-1,-1,-1,-1), nthreads=nthreads, res=out)
    return out

def mcm02_ducc_tri(spec, lmax):
    out= np.empty((spec.shape[0],5,((lmax+1)*(lmax+2))//2),dtype=np.float32)
    ducc0.misc.experimental.coupling_matrix_spin0and2_tri(spec[:,:,:], lmax, (0,1,2,3), (0,1,2,3,4), nthreads=nthreads, res=out)
    return out

def mcmpm_ducc_tri(spec, lmax):
    out= np.empty((spec.shape[0],2,((lmax+1)*(lmax+2))//2),dtype=np.float32)
    ducc0.misc.experimental.coupling_matrix_spin0and2_tri(spec[:,3:,:], lmax, (0,0,0,0), (-1,-1,-1,0,1), nthreads=nthreads, res=out)
    return out

def mcm02_pure_ducc(spec, lmax):
    res = np.empty((nspec, 4, lmax+1, lmax+1), dtype=np.float32)
    return ducc0.misc.experimental.coupling_matrix_spin0and2_pure(spec, lmax, nthreads=nthreads, res=res)

# Modified version of ducc0 mcm_bench.py
class Benchmark(object):
    def __init__(self,lmax, bin_edges = None):
        self.lmax = lmax
        # number of spectra to process simultaneously
        nspec=1
        print()
        print("Mode coupling matrix computation comparison")
        print(f"nspec={nspec}, lmax={lmax}, nthreads={nthreads}")
        # we generate the spectra up to 2*lmax+1 to use all Wigner 3j symbols
        # but this could also be lower.
        seed = 1
        np.random.seed(seed)
        cls = np.random.normal(size=(2*lmax+1,))
        self.spec = np.repeat(cls[None, :], repeats=4, axis=0)[None,...]


    def get_mcm(self,code,spin=0,bin_edges=None,bin_weights=None):

        a = time()
        nbins = len(bin_edges)-1
        if code=='ducc':
            if spin==0:
                ducc = mcm00_ducc_tri(self.spec[:,0,:], self.lmax)
                mcm = tri2full(ducc, self.lmax)[:,0,:,:][0]
            elif spin==2:
                duccpm = mcmpm_ducc_tri(self.spec, self.lmax)
                mcmi = tri2full(duccpm, self.lmax)[0]
            if bin_edges is not None:
                if spin==0:
                    mcm = utils.bin_square_matrix(mcm,bin_edges,self.lmax,bin_weights=bin_weights)
                elif spin==2:
                    mcm = np.zeros((2,nbins,nbins))
                    for i in range(2):
                        mcm[i] = utils.bin_square_matrix(mcmi[i],bin_edges,self.lmax,bin_weights=bin_weights)
            else:
                if spin==0:
                    mcm = mcm[2:,2:]
                elif spin==2:
                    mcm = mcm[:,2:,2:]

                print(mcm.shape)
        elif code=='wiggle':
            if spin==0:
                mcm = pywiggle.get_coupling_matrix_from_mask_cls(self.spec[0,0],self.lmax,spintype='00',bin_edges = bin_edges,bin_weights = bin_weights)
            elif spin==2:
                mcm1, g = pywiggle.get_coupling_matrix_from_mask_cls(self.spec[0,0],self.lmax,spintype='++',bin_edges = bin_edges,bin_weights = bin_weights, return_obj=True)
                mcm1 = mcm1
                mcm2 = g.get_coupling_matrix_from_ids('m1','m1',spintype='--',bin_weight_id='b1',beam_id1=None,beam_id2=None)
                mcm = np.zeros((2,*mcm1.shape))
                mcm[0] = mcm1
                mcm[1] = mcm2

            if bin_edges is None:
                if spin==0:
                    mcm = mcm[2:,2:]
                elif spin==2:
                    mcm = mcm[:,2:,2:]
                
        else:
            raise ValueError

        etime = time()-a

        return mcm,etime

@pytest.mark.parametrize("lmax", [1024, 2048, 4000])#, 4096, 8192, 16384])
def test_ducc0_comparison(lmax):

    b = Benchmark(lmax=lmax)
    bin_edges = np.arange(40,b.lmax,40)


    for spin in [0,2]:
        times = {}
        mcm_s0s = {}
        bcode = 'ducc'
        codes = ['wiggle']
        for code in [bcode,]+codes:
            mcm_s0s[code], times[code] = b.get_mcm(code,spin=spin,bin_edges = bin_edges)
            print(f"{code} time: {(times[code]*1000):.1f} ms")
            if code!=bcode:
                l2 = ducc0.misc.l2error(mcm_s0s[code],mcm_s0s[bcode])
                print(f"L2 error between {code} and {bcode} solutions: {l2}")
                for offset in range(3):
                    if spin==2:
                        for i in range(2):
                            np.testing.assert_allclose(np.diagonal(mcm_s0s[code][i],offset), np.diagonal(mcm_s0s[bcode][i],offset), rtol=1e-6)
                    elif spin==0:
                        np.testing.assert_allclose(np.diagonal(mcm_s0s[code],offset), np.diagonal(mcm_s0s[bcode],offset), rtol=1e-6)
                np.testing.assert_allclose(l2, 0., atol=1e-7)

