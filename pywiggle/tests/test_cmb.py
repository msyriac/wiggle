import matplotlib
matplotlib.use('Agg')
import numpy as np
from pixell import enmap, enplot, curvedsky as cs, utils, bench,reproject
import matplotlib.pyplot as plt
from pywiggle import utils as wutils
import pywiggle
import io,sys
import healpy as hp
from collections import defaultdict

def print_keys_tree(d, indent=0):
    for key, value in d.items():
        print("  " * indent + str(key))
        if isinstance(value, dict):
            print_keys_tree(value, indent + 1)

nside = 128
lmax = 2*nside

res = 16.0 / 60. *(128/nside) # deg
beam = res * 60. * 2 #arcmin
shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res))

area_deg2 = 4000.
apod_deg = 10.0
smooth_deg = 10.0
radius_deg = np.sqrt(area_deg2 / np.pi)

bin_edges = np.arange(20,lmax,20)


# Load CMB Cls ---
ps, ells = wutils.load_test_spectra()
assert ps.shape == (3, 3, len(ells))

alm = cs.rand_alm(ps, lmax=lmax)

imap = cs.alm2map(alm, enmap.empty((3,)+shape, wcs,dtype=np.float32))
_,mask = wutils.get_mask(nside,shape,wcs,radius_deg,apod_deg,smooth_deg=smooth_deg)
mask_alm = cs.map2alm(mask,lmax=2*lmax,spin=0)

imap = imap * mask
oalms = cs.map2alm(imap,lmax=lmax,spin=[0,2])

#for i in range(3): wutils.hplot(imap[i],f'imap_{i}',grid=True,ticks=20)
#wutils.hplot(mask,f'mask',grid=True,ticks=20)


_, pureB = pywiggle.get_pure_EB_alms(imap[1], imap[2], mask,lmax=lmax,masked_on_input=True)



ret = pywiggle.get_powers(oalms,oalms, mask_alm,return_theory_filter=True,lmax=lmax,bin_edges=bin_edges)

oalms[2] = pureB.copy()
ret_pure = pywiggle.get_powers(oalms,oalms, mask_alm,return_theory_filter=True,lmax=lmax,bin_edges=bin_edges)
