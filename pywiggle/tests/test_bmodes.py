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


def test_recover_tensor_Bmode():
    import pymaster as nmt
    # Sim config

    res = 16.0 / 60. # deg
    beam = res * 60. * 2 #arcmin
    nside = 256
    shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res))
    lmax = 3*nside
    mlmax = 2*nside
    hpixdiv = 1
    cardiv = 1
    nsims = 3

    area_deg2 = 4000.
    apod_deg = 10.0
    smooth_deg = 10.0
    radius_deg = np.sqrt(area_deg2 / np.pi)
    radius_rad = np.deg2rad(radius_deg)

    # Load CMB Cls ---
    with bench.show("camb"):
        ps = wutils.get_camb_spectra(lmax=lmax)
    ells = np.arange(ps[0,0].size)
    # ps, ells = wutils.load_test_spectra()
    assert ps.shape == (3, 3, len(ells))

    theory = wutils.get_cldict(ps)
    
    def compute_master(f_a, f_b, wsp):
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        return cl_decoupled
    

    maskh, mask = wutils.get_mask(nside,shape,wcs,radius_deg,apod_deg,smooth_deg)
    # wutils.hplot(mask,'mask',grid=True,colorbar=True,downgrade=4,ticks=30)
    
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


        
        f2yp = nmt.NmtField(maskh, [Qh, Uh], purify_e=False, purify_b=True,n_iter=0,lmax=int(mlmax/hpixdiv),lmax_mask=int(mlmax/hpixdiv))
        # Healpix Namaster Purified
        w_yp = nmt.NmtWorkspace.from_fields(f2yp, f2yp, b)
        cl_yp_nmt = compute_master(f2yp, f2yp, w_yp)


    
        pureE, pureB = pywiggle.get_pure_EB_alms(Q, U, mask,lmax=mlmax/cardiv)

        ialms = np.zeros((2,oalm[0].size),dtype=np.complex128)
        ialms[0]  = oalm[0]
        ialms[1] = wutils.change_alm_lmax(pureB,mlmax) # impure E, pure B
        w = pywiggle.Wiggle(mlmax, bin_edges=bin_edges,verbose=False)
        w.add_mask('m', mask_alm)
        ret = w.get_powers(ialms,ialms, 'm',return_theory_filter=True if i==0 else False,pure_B = True)
        if i==0:
            bth_pure = pywiggle.get_binned_theory(ret,theory)

        cl_EE = ret['EE']['Cls']
        cl_bb_wig_p = ret['BB']['Cls'].copy()

        ialms[0]  = oalm[0]
        ialms[1] = oalm[1] # impure E, impure B
        w = pywiggle.Wiggle(mlmax, bin_edges=bin_edges,verbose=False)
        w.add_mask('m', mask_alm)
        ret = w.get_powers(ialms,ialms, 'm',return_theory_filter=True if i==0 else False,pure_B = False)
        if i==0:
            bth = pywiggle.get_binned_theory(ret,theory)
        icl_EE = ret['EE']['Cls']
        cl_bb_wig_i = ret['BB']['Cls'].copy()

        results["Namaster BB pure decoupled (healpix)"].append(cl_yp_nmt[3].copy())
        results["Wiggle BB pure decoupled (CAR)"].append(cl_bb_wig_p.copy())
        results["Wiggle BB impure decoupled (CAR)"].append(cl_bb_wig_i.copy())

        from orphics import io
        pl = io.Plotter('Cell')
        print(bth_pure['BB'])
        pl.add(leff,bth['BB']*leff*(leff+1.)/2./np.pi,label='binned BB theory',marker='x',color='r',ls='none')
        pl.add(leff+5,bth_pure['BB']*leff*(leff+1.)/2./np.pi,label='binned BB theory pure',marker='x',color='b',ls='none')
        pl.done('bb.png')
        sys.exit()

        
        


    means = {}
    errs = {}

    for label, values in results.items():
        stacked = np.stack(values)
        means[label] = np.mean(stacked, axis=0)
        errs[label] = np.sqrt(np.var(stacked, axis=0, ddof=1)/nsims)
        
    # Compute power spectrum and compare ---
    input_bb = ps[2, 2]
    ls = np.arange(input_bb.size)

    
    plt.figure()
    plt.plot(ls, input_bb*ls*(ls+1)/2./np.pi, label='Input BB',ls='--')
    plt.plot(ells, bb_orig*ells*(ells+1)/2./np.pi, label='Full-sky unmasked BB power',alpha=0.5)
    plt.plot(els, bb_masked*els*(els+1)/2./np.pi, label='Masked BB power divided by mean(mask**2)')
    for i,key in enumerate(results.keys()):
        print(key)
        print(means)
        plt.errorbar(leff+i*3,means[key]*leff*(leff+1.)/2./np.pi,yerr=errs[key]*leff*(leff+1.)/2./np.pi,label=key,ls='none',marker='o')
    plt.plot(leff,bth['BB']*leff*(leff+1.)/2./np.pi,label='binned BB theory',marker='x',color='r',ls='none')
    plt.plot(leff+5,bth_pure['BB']*leff*(leff+1.)/2./np.pi,label='binned BB theory pure',marker='x',color='b',ls='none')
    
    plt.xlim(2, 300)
    plt.yscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$D_\ell^{BB}$')
    plt.legend()
    plt.title(f'B-mode recovery test ({area_deg2:.0f} deg$^2$ mask)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bmodes.png',dpi=200)
    plt.close()

    plt.figure()
    for i,key in enumerate(results.keys()):
        if key!="Wiggle BB impure decoupled (CAR)":
            plt.plot(leff+i*3,errs[key]/errs["Wiggle BB impure decoupled (CAR)"],label=key,marker='o')
    plt.ylabel('$\\sigma(C_{\\ell}^{\\rm pure})/\\sigma(C_{\\ell}^{\\rm impure})$')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.yscale('log')
    plt.axhline(y=1)
    # plt.ylim(0.05,20.0)
    plt.xlim(2, 300)
    plt.savefig('berrrat.png',dpi=200)
    plt.close()

