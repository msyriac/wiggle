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

nsims = 5


def unpack_cls(ret):
    include_keys = ['TT','EE','TE','EB','BE','BB']
    return {k: v['Cls'] for k, v in ret.items() if k in include_keys}
    # out = {k: v['Cls'] for k, v in d.items()}
    # # TT and its theory filter
    # tt = ret['TT']['Cls']
    
    # # EE and its theory filter
    # te = ret['TE']['Cls']
    
    # # EE, BB and the theory filter for (EE, EB, BE, BB)
    # ee = ret['EE']['Cls']
    # bb = ret['BB']['Cls']

def unpack_theory(ret,cltt,clte,clee,clbb,cleb=None):
    ret_th = {}
    tt_th = ret['TT']['Th']
    lmax = tt_th.shape[1]

    ret_th['TT'] = tt_th @ cltt[:lmax]
    
    te_th = ret['TE']['Th']
    ret_th['TE'] = te_th @ clte[:lmax]
    
    pol_th = ret['ThPol']
    nbins = tt_th.shape[0]
    if cleb is None: cleb = clbb*0.
    clpol = np.concatenate( [clee[:lmax], cleb[:lmax], cleb[:lmax], clbb[:lmax] ] )
    bpol = pol_th @ clpol

    rlist = ['EE','EB','BE','BB']
    start = 0
    step = nbins
    for i,spec in enumerate(rlist):
        end = start + step
        ret_th[spec] = bpol[start:end]
        if ret_th[spec].shape[0]!=step: raise ValueError
        start = end
    return ret_th


s = stats.Stats()

for i in range(nsims):
    alm = cs.rand_alm(ps, lmax=lmax)

    imap = cs.alm2map(alm, enmap.empty((3,)+shape, wcs,dtype=np.float32))
    _,mask = wutils.get_mask(nside,shape,wcs,radius_deg,apod_deg,smooth_deg=smooth_deg)
    mask_alm = cs.map2alm(mask,lmax=2*lmax,spin=0)

    imap = imap * mask
    oalms = cs.map2alm(imap,lmax=lmax,spin=[0,2])

    if i==0:
        for j in range(3): wutils.hplot(imap[j],f'imap_{j}',grid=True,ticks=20)
        wutils.hplot(mask,f'mask',grid=True,ticks=20)

    # Purify B
    _, pureB = pywiggle.get_pure_EB_alms(imap[1], imap[2], mask,lmax=lmax,masked_on_input=True)


    # Get impure power
    ret = pywiggle.get_powers(oalms,oalms, mask_alm,return_theory_filter=True,lmax=lmax,bin_edges=bin_edges)
    bcls = unpack_cls(ret)
    if i==0:
        bth = unpack_theory(ret,ps[0,0],ps[0,1],ps[1,1],ps[2,2])

    # Get pure power
    oalms[2] = pureB.copy()
    ret_pure = pywiggle.get_powers(oalms,oalms, mask_alm,return_theory_filter=True,lmax=lmax,bin_edges=bin_edges)
    bcls_pure = unpack_cls(ret_pure)
    if i==0:
        bth_pure = unpack_theory(ret_pure,ps[0,0],ps[0,1],ps[1,1],ps[2,2])
