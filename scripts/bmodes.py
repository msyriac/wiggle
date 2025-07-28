#%%
%load_ext autoreload
%autoreload 2
#%%
import pywiggle 
from pywiggle import utils as wutils
from pixell import enmap, utils as u, curvedsky as cs
import healpy as hp
import numpy as np
from orphics import io as oio, maps
import pymaster as nmt
import os,sys
from collections import defaultdict
#%%

beam_arcmin = 16 # must be factors of 2 smaller or larger
noise = 1. # uK-arcmin; or None

hpixdiv = 2 # divide healpix mlmax by this
cardiv = 2  # divide CAR mlmax by this

area_deg2 = 4000.
apod_deg = 10.0


beam_deg = beam_arcmin / 60. # deg
res_deg = beam_deg  / 2. # for CAR pixels
nside = int(512  *  16 / (beam_arcmin))



shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res_deg))

lmax = 3*nside # lmax out to which Cls are populated
mlmax = 2*nside # maximum lmax for SHTs

radius_deg = np.sqrt(area_deg2 / np.pi)
radius_rad = np.deg2rad(radius_deg)

ps = wutils.get_camb_spectra(lmax=lmax)

def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled



maskh, mask = wutils.get_mask(nside,shape,wcs,radius_deg,apod_deg)
oio.hplot(mask,'mask',grid=True,colorbar=True,downgrade=4,ticks=30)


# Mode decoupling
mask_alm = cs.map2alm(mask, lmax=2 * mlmax)


# Binning
b = nmt.NmtBin.from_nside_linear((nside*2/3./hpixdiv)+0.5, 16)
leff = b.get_effective_ells()
nbins = leff.size
bin_edges = []
for i in range(nbins):
    bin_edges.append(b.get_ell_min(i))
bin_edges.append(b.get_ell_max(nbins-1))
bcents = leff 
w2 = wutils.wfactor(2,mask)


nsims = 40
results = defaultdict(list)


def get_cmb_sim(ps,beam,noise,seed):
    np.random.seed((1,seed))
    alm = cs.rand_alm(ps, lmax=lmax)
    ls = np.arange(ps[0,0].size)
    for i in range(3):
        alm[i] = cs.almxfl(alm[i],maps.gauss_beam(beam,ls))
    polmap = cs.alm2map(alm[1:], enmap.empty((2,)+shape, wcs,dtype=np.float32), spin=2)  # only Q,U

    hmap = hp.alm2map(alm,nside,pol=True)
    Qh = hmap[1]
    Uh = hmap[2]
    Q = polmap[0].copy()
    U = polmap[1].copy()

    nc1 = maps.white_noise(shape,wcs,noise_muK_arcmin=noise,seed=(2,seed),ipsizemap=None,div=None,nside=None)
    nc2 = maps.white_noise(shape,wcs,noise_muK_arcmin=noise,seed=(3,seed),ipsizemap=None,div=None,nside=None)

    nh1 = maps.white_noise(None,None,noise_muK_arcmin=noise,seed=(2,seed),ipsizemap=None,div=None,nside=nside)
    nh2 = maps.white_noise(None,None,noise_muK_arcmin=noise,seed=(3,seed),ipsizemap=None,div=None,nside=nside)

    return Q+nc1, U+nc1, Q+nc2, U+nc2, Qh+nh1, Uh+nh1, Qh+nh2, Uh+nh2, alm
    
    
for i in range(nsims):
    print(i)

    Q1,U1, Q2, U2, Qh1,Uh1, Qh2, Uh2, alm = get_cmb_sim(ps,beam_deg/60.,noise,i)
    
    if i==0:
        bb_orig = cs.alm2cl(alm[2])
        ells = np.arange(bb_orig.size)
        ee_orig = cs.alm2cl(alm[1])

    

    masked1 = enmap.enmap([Q1*mask,U1*mask],Q1.wcs)
    masked2 = enmap.enmap([Q2*mask,U2*mask],Q2.wcs)




    oalm1 = cs.map2alm(masked1,lmax=mlmax,spin=2)
    oalm2 = cs.map2alm(masked2,lmax=mlmax,spin=2)
    if i==0:
        bb_masked = cs.alm2cl(oalm1[1],oalm2[1])
        bb_masked = bb_masked / w2
        els = np.arange(bb_masked.size)




    f2yp = nmt.NmtField(maskh, [Qh, Uh], purify_e=False, purify_b=True,n_iter=0,lmax=int(mlmax/hpixdiv),lmax_mask=int(mlmax/hpixdiv))

    # Healpix Namaster Purified
    w_yp = nmt.NmtWorkspace.from_fields(f2yp, f2yp, b)
    cl_yp_nmt = compute_master(f2yp, f2yp, w_yp)



    pureE, pureB = pywiggle.get_pure_EB_alms(Q, U, mask,lmax=mlmax/cardiv)

    ialms = np.zeros((2,oalm[0].size),dtype=np.complex128)
    ialms[0]  = oalm[0]
    ialms[1] = maps.change_alm_lmax(pureB,mlmax) # impure E, pure B
    w = pywiggle.Wiggle(mlmax, bin_edges=bin_edges)
    w.add_mask('m', mask_alm)
    ret = w.decoupled_cl(ialms,ialms, 'm',return_theory_filter=False,pure_B = True)

    cl_EE = ret['EE']['Cls']
    cl_bb_wig_p = ret['BB']['Cls'].copy()

    ialms[0]  = oalm[0]
    ialms[1] = oalm[1] # impure E, impure B
    w = pywiggle.Wiggle(mlmax, bin_edges=bin_edges)
    w.add_mask('m', mask_alm)
    ret = w.decoupled_cl(ialms,ialms, 'm',return_theory_filter=False,pure_B = False)
    icl_EE = ret['EE']['Cls']
    cl_bb_wig_i = ret['BB']['Cls'].copy()

    results["Namaster BB pure decoupled (healpix)"].append(cl_yp_nmt[3].copy())
    results["Wiggle BB pure decoupled (CAR)"].append(cl_bb_wig_p.copy())
    results["Wiggle BB impure decoupled (CAR)"].append(cl_bb_wig_i.copy())




means = {}
errs = {}

for label, values in results.items():
    stacked = np.stack(values)
    means[label] = np.mean(stacked, axis=0)
    errs[label] = np.sqrt(np.var(stacked, axis=0, ddof=1)/nsims)

# Compute power spectrum and compare ---
# bpow_nam = nam_bb / w2
# bpow_nam_pixell = nam_bb_pixell / w2
# bpow = cs.alm2cl(pureB) / w2
# epow = cs.alm2cl(pureE) / w2
input_bb = ps[2, 2]
# input_ee = ps[1, 1]
ls = np.arange(input_bb.size)
# ell = np.arange(bpow.size)


plt.figure()
# ell = np.arange(len(bpow))
# elln = np.arange(len(bpow_nam))
plt.plot(ls, input_bb, label='Input BB',ls='--')
plt.plot(ells, bb_orig, label='Full-sky unmasked BB power',alpha=0.5)
plt.plot(els, bb_masked, label='Masked BB power divided by mean(mask**2)')
for i,key in enumerate(results.keys()):
    print(key)
    print(means)
    plt.errorbar(leff+i*3,means[key],yerr=errs[key],label=key,ls='none',marker='o')
# plt.plot(elsh, bb_maskedh, label='Masked BB power divided by mean(mask**2), lmax/2')
# plt.plot(elln, bpow_nam, label='Recovered pure B (Nmt)')
# plt.plot(elln, bpow_nam_pixell, label='Recovered pure B (Nmt; CAR)')
# plt.plot(ell, bpow, label='Recovered pure B')
# plt.plot(bcents,cl_BB, label = 'Decoupled pure B', marker='d', ls='none')
# plt.plot(bcents,ncl_BB, label = 'Decoupled wiggle, pure B Nmt', marker='d', ls='none')
# plt.plot(bcents,icl_BB, label = 'Decoupled impure B', marker='o', ls='none')
# plt.plot(leff,cl_p_bb, label = 'Decoupled pure B (Nmt)', marker='x', ls='none')

# plt.plot(oell, obpow, label='Recovered pure B (masked on input)')
plt.xlim(2, 300)
plt.yscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{BB}$')
plt.legend()
plt.title(f'B-mode recovery test ({area_deg2:.0f} deg$^2$ mask)')
plt.grid(True)
plt.tight_layout()
plt.savefig('bmodes.png',dpi=200)

pl = oio.Plotter('rCl',ylabel='$\\sigma(C_{\\ell}^{\\rm pure})/\\sigma(C_{\\ell}^{\\rm impure})$',xyscale='linlog')
for i,key in enumerate(results.keys()):
    if key!="Wiggle BB impure decoupled (CAR)":
        pl.add(leff+i*3,errs[key]/errs["Wiggle BB impure decoupled (CAR)"],label=key,marker='o')
pl.hline(y=1)
pl._ax.set_ylim(0.3,100.0)
pl._ax.set_xlim(2, 300)
pl.done('berrrat.png')

# plt.figure()
# ell = np.arange(len(epow))
# plt.plot(ls, input_ee, label='Input EE',ls='--')
# plt.plot(ells, ee_orig, label='Full-sky unmasked EE power',alpha=0.5)
# plt.plot(els, ee_masked, label='Masked EE power / mean(mask**2)')
# plt.plot(ell, epow, label='Recovered pure E')
# # plt.plot(oell, oepow, label='Recovered pure E (masked on input)')
# # plt.plot(bcents,icl_EE, label = 'Decoupled impure E', marker='o', ls='none')
# # plt.plot(bcents+1,cl_EE, label = 'Decoupled impure E, purified B', marker='o', ls='none')
# # plt.plot(leff,cl_np_ee, label = 'Decoupled impure E (Nmt)', marker='x', ls='none')
# plt.xlim(2, 300)
# plt.yscale('log')
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$C_\ell^{EE}$')
# plt.legend()
# plt.title(f'E-mode recovery test ({area_deg2:.0f} deg$^2$ mask)')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('emodes.png',dpi=200)


# # Compute power spectrum and compare ---
# plt.figure()
# plt.plot(ls, input_ee, label='Input EE',ls='--')
# plt.plot(ells, ee_orig, label='Full-sky unmasked EE power',alpha=0.5)
# plt.plot(ell, bpow, label='Recovered pure B')
# plt.plot(bcents,cl_BB, label = 'Decoupled pure B', marker='d', ls='none')
# print(cl_BB)
# plt.xlim(2, 300)
# plt.yscale('log')
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$C_\ell^{BB}$')
# plt.legend()
# plt.title(f'B-mode recovery test ({area_deg2:.0f} deg$^2$ mask)')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('bmodes_alone.png',dpi=200)


# %%
