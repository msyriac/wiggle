import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from pixell import enmap, enplot, curvedsky as cs, bench,reproject
import matplotlib.pyplot as plt
from pywiggle import utils as wutils
import pywiggle
import io,sys
import healpy as hp
from collections import defaultdict
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from orphics import io, stats, mpi, maps
import utils


nside = 512
lmax = 2*nside
tlmax = 3*nside

res = 16.0 / 60. *(128/nside) # deg
beam = res * 60. * 2 #arcmin
shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res))

area_deg2 = 4000.
apod_deg = 10.0
smooth_deg = 10.0
radius_deg = np.sqrt(area_deg2 / np.pi)

bin_edges = np.append(np.append(np.geomspace(2,200,10),np.arange(240,2400,60)),np.arange(2550,8000,150))
bin_edges = bin_edges[bin_edges<lmax]


nsims = 240
comm,rank,my_tasks = mpi.distribute(nsims)

bshow = lambda x: (bench.show(x) if rank==0 else io.no_context())

with bshow("camb"):
    ps = wutils.get_camb_spectra(lmax=tlmax)
ells = np.arange(ps[0,0].size)
assert ps.shape == (3, 3, len(ells))


def unpack_cls(ret):
    include_keys = ['TT','EE','TE','EB','BE','BB','TB']
    return {k: v['Cls'] for k, v in ret.items() if k in include_keys}


theory = wutils.get_cldict(ps)

s = stats.Statistics(comm)

for i,task in enumerate(my_tasks):
    print(f"Rank {rank} doing {i+1}/{len(my_tasks)}, task number {task}.")
    with bshow("sim"):
        alm = cs.rand_alm(ps, lmax=tlmax, seed=task)
        imap = cs.alm2map(alm, enmap.empty((3,)+shape, wcs,dtype=np.float32))
    if i==0:
        with bshow("mask"):
            try:
                mask = enmap.read_map(f'mask_{nside}.fits')
                mask_alm = hp.read_alm(f'mask_alm_{nside}.fits')
            except:
                mask = maps.circular_mask(shape,wcs,(0.,0.),radius_deg,apod_deg,smooth_deg,lmax=tlmax)
                # _,mask = wutils.get_mask(nside,shape,wcs,radius_deg,apod_deg,smooth_deg=smooth_deg)
                mask_alm = cs.map2alm(mask,lmax=2*lmax,spin=0)
                if rank==0:
                    enmap.write_map(f'mask_{nside}.fits',mask)
                    hp.write_alm(f'mask_alm_{nside}.fits',mask_alm,overwrite=True)
                
        with bshow("purify init"):
            ebp = pywiggle.EBPurifier(mask,lmax)

    omap = imap * mask
    with bshow("map2alm"):
        oalms = cs.map2alm(omap,lmax=lmax,spin=[0,2])

    if i==0 and lmax<512 and rank==0:
        for j in range(3): wutils.hplot(imap[j],f'imap_{j}',grid=True,ticks=20)
        wutils.hplot(mask,f'mask',grid=True,ticks=20)

    with bshow("purify"):
        # Purify B
        _, pureB = ebp.project(omap[1], omap[2],masked_on_input=True)


    # Get impure power
    with bshow("wiggle"):
        ret = pywiggle.get_powers(oalms,oalms, mask_alm,return_theory_filter=True,lmax=lmax,bin_edges=bin_edges)
    bcls = unpack_cls(ret)
    for spec in ['TT','EE','TE','BB','EB','TB']:
        s.add(spec,bcls[spec])
    
    if i==0:
        bth = pywiggle.get_binned_theory(ret,theory)

    # Get pure power
    oalms[2] = pureB.copy()
    with bshow("wiggle"):
        ret_pure = pywiggle.get_powers(oalms,oalms, mask_alm,return_theory_filter=True,lmax=lmax,bin_edges=bin_edges,pure_B=True)
    bcls_pure = unpack_cls(ret_pure)
    s.add('BB pure',bcls_pure['BB'])
    s.add('TB pure',bcls_pure['TB'])
    s.add('EB pure',bcls_pure['EB'])
    if i==0:
        bth_pure = pywiggle.get_binned_theory(ret_pure,theory)

s.allreduce()
s.save_reduced(f'stats_{nside}.npz')

if rank==0:
    io.save_dict(f'bth_{nside}.h5',bth)
    io.save_dict(f'bth_pure_{nside}.h5',bth_pure)
    io.save_dict(f'theory_{nside}.h5',theory)
    np.savetxt(f'bin_edges_{nside}.txt',bin_edges)
    utils.analyze(s,bth,bth_pure,theory,bin_edges,nside,do_pure=True)
    utils.analyze(s,bth,bth_pure,theory,bin_edges,nside,do_pure=False)

    print("Done.")
