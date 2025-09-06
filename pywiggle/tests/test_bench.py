import os
try:
    nthreads = int(os.environ["OMP_NUM_THREADS"])
except:
    import multiprocessing
    nthreads = multiprocessing.cpu_count()

import numpy as np
import healpy as hp
import pytest
import pywiggle
from pywiggle import utils
import warnings
import ducc0    

def test_ducc0_comparison():

    for lmax in [512, 1024, 2048, 4096]:
        for binned in [False,True]:
            b = utils.Benchmark(lmax=lmax,nthreads=nthreads,verbose=True)
            if binned:
                bin_edges = np.arange(40,b.lmax,40)
                print("==============")
                print(lmax, " binned")
                print("==============")
                dtol = 1e-6
            else:
                bin_edges = None
                print("==============")
                print(lmax, " unbinned")
                print("==============")
                dtol = 1e-4


            for spin in [0,2]:
                times = {}
                mcm_s0s = {}
                bcode = 'ducc'
                codes = ['wiggle']
                for code in [bcode,]+codes:
                    mcm_s0s[code], times[code] = b.get_mcm(code,spin=spin,bin_edges = bin_edges)
                    utils.cprint(f"spin {spin}, {code} time: {(times[code]*1000):.1f} ms",color='g')
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

