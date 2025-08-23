import matplotlib
matplotlib.use('Agg')
import numpy as np
from pixell import enmap, enplot, curvedsky as cs, utils, bench,reproject
import matplotlib.pyplot as plt
from pywiggle import utils as wutils
import pywiggle
import io,sys
import pymaster as nmt
import healpy as hp
from orphics import maps, io as oio
from collections import defaultdict






def test_recover_tensor_Bmode():
    # Sim config ---

    res = 16.0 / 60. # deg
    beam = res * 60. * 2 #arcmin
    nside = 256
    shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res))
    lmax = 3*nside
    mlmax = 2*nside
    hpixdiv = 2
    cardiv = 2
    nsims = 10

    area_deg2 = 4000.
    apod_deg = 10.0
    smooth_deg = 10.0
    radius_deg = np.sqrt(area_deg2 / np.pi)
    radius_rad = np.deg2rad(radius_deg)

    # Load CMB Cls ---
    ps, ells = wutils.load_test_spectra()
    assert ps.shape == (3, 3, len(ells))

    def compute_master(f_a, f_b, wsp):
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        return cl_decoupled
    

    # Make apodized circular mask
    # modrmap = enmap.modrmap(shape,wcs)
    # bmask = enmap.zeros(shape,wcs)
    # bmask[modrmap<radius_rad] = 1
    # dist      = enmap.distance_transform(bmask == 1)
    # mask = np.clip(dist / np.deg2rad(apod_deg), 0, 1)
    # mask = 0.5 - 0.5*np.cos(np.pi * mask)   # raised-cosine window
    # maskh = reproject.map2healpix(mask, nside=nside , method="spline", order=1, extensive=False)

    maskh, mask = wutils.get_mask(nside,shape,wcs,radius_deg,apod_deg,smooth_deg)#hp_file="/home1/mathm/repos/wiggle/analysis_mask_apo10_C1_nside256.fits")
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

    # bin_edges = np.append([2,10,20], np.arange(40,lmax,10))
    bcents = leff #(bin_edges[1:]+bin_edges[:-1])/2.
    w2 = wutils.wfactor(2,mask)
    
    
    results = defaultdict(list)
    
    for i in range(nsims):
        print(i)
        # Simulate polarization map ---
        np.random.seed(i)
        alm = cs.rand_alm(ps, lmax=lmax)
        
        if i==0:
            bb_orig = cs.alm2cl(alm[2])
            ells = np.arange(bb_orig.size)
            ee_orig = cs.alm2cl(alm[1])

        # ls = np.arange(ps[0,0].size)
        # for j in range(3):
        #     alm[j] = cs.almxfl(alm[j],maps.gauss_beam(beam,ls))
            
            
        polmap = cs.alm2map(alm[1:], enmap.empty((2,)+shape, wcs,dtype=np.float32), spin=2)  # only Q,U
        
        hmap = hp.alm2map(alm,nside,pol=True)
        Qh = hmap[1]
        Uh = hmap[2]
        Q = polmap[0].copy()
        U = polmap[1].copy()


        masked = polmap*mask




        oalm = cs.map2alm(masked,lmax=mlmax,spin=2)
        if i==0:
            bb_masked = cs.alm2cl(oalm[1],oalm[1])
            bb_masked = bb_masked / w2
            els = np.arange(bb_masked.size)


        # f2yp = nmt.NmtField(mask, [Q,U], purify_e=False, purify_b=True,n_iter=0,wcs=Q.wcs,lmax=mlmax,lmax_mask=mlmax)
        # nam_map = f2yp.get_maps().reshape((2,Q.shape[0],Q.shape[1]))
        # print(Q.shape,nam_map.shape)
        # palm = cs.map2alm(enmap.enmap(nam_map,Q.wcs),spin=2,lmax=mlmax)
        # nam_bb_pixell = cs.alm2cl(palm[1],palm[1])
        
        
        f2yp = nmt.NmtField(maskh, [Qh, Uh], purify_e=False, purify_b=True,n_iter=0,lmax=int(mlmax/hpixdiv),lmax_mask=int(mlmax/hpixdiv))
        # nam_map = f2yp.get_maps()
        # halm = hp.map2alm([Qh*0,nam_map[0],nam_map[1]],pol=True,iter=0,lmax=mlmax)
        # nam_bb = cs.alm2cl(halm[2],halm[2])

        # Healpix Namaster Purified
        w_yp = nmt.NmtWorkspace.from_fields(f2yp, f2yp, b)
        cl_yp_nmt = compute_master(f2yp, f2yp, w_yp)


    
        pureE, pureB = pywiggle.get_pure_EB_alms(Q, U, mask,lmax=mlmax/cardiv)

        ialms = np.zeros((2,oalm[0].size),dtype=np.complex128)
        ialms[0]  = oalm[0]
        ialms[1] = maps.change_alm_lmax(pureB,mlmax) # impure E, pure B
        w = pywiggle.Wiggle(mlmax, bin_edges=bin_edges)
        w.add_mask('m', mask_alm)
        ret = w.get_powers(ialms,ialms, 'm',return_theory_filter=False,pure_B = True)

        cl_EE = ret['EE']['Cls']
        cl_bb_wig_p = ret['BB']['Cls'].copy()

        ialms[0]  = oalm[0]
        ialms[1] = oalm[1] # impure E, impure B
        w = pywiggle.Wiggle(mlmax, bin_edges=bin_edges)
        w.add_mask('m', mask_alm)
        ret = w.get_powers(ialms,ialms, 'm',return_theory_filter=False,pure_B = False)
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
    pl._ax.set_ylim(0.05,20.0)
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
    
