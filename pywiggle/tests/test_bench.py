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
        if bin_edges is not None:
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
                    mcm = mcmi[:,2:,2:]

                print(mcm.shape)
        elif code=='wiggle':
            if spin==0:
                mcm = pywiggle.get_coupling_matrix_from_mask_cls(self.spec[0,0],self.lmax,spintype='00',bin_edges = bin_edges,bin_weights = bin_weights)
            elif spin==2:
                mcm1, g = pywiggle.get_coupling_matrix_from_mask_cls(self.spec[0,0],self.lmax,spintype='+',bin_edges = bin_edges,bin_weights = bin_weights, return_obj=True)
                mcm1 = mcm1
                mcm2 = g.get_coupling_matrix_from_ids('m1','m1',spintype='-',bin_weight_id='b1',beam_id1=None,beam_id2=None)
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

def test_ducc0_comparison():

    for lmax in [1024,2048,4000]:
        for binned in [False,True]:
            b = Benchmark(lmax=lmax)
            if binned:
                bin_edges = np.arange(40,b.lmax,40)
                print(lmax, " binned")
                dtol = 1e-6
            else:
                bin_edges = None
                print(lmax, " unbinned")
                dtol = 1e-3


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
                                    np.testing.assert_allclose(np.diagonal(mcm_s0s[code][i],offset), np.diagonal(mcm_s0s[bcode][i],offset), rtol=dtol)
                            elif spin==0:
                                np.testing.assert_allclose(np.diagonal(mcm_s0s[code],offset), np.diagonal(mcm_s0s[bcode],offset), rtol=dtol)
                        np.testing.assert_allclose(l2, 0., atol=1e-7)

