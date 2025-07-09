import numpy as np
from pixell import enmap, enplot, curvedsky as cs, utils, bench,reproject
import matplotlib.pyplot as plt
import pywiggle 
import pkgutil
import io
import pymaster as nmt
import healpy as hp

def wfactor(n,mask,sht=True,pmap=None,equal_area=False):
    """
    Copied from msyriac/orphics/maps.py
    
    Approximate correction to an n-point function for the loss of power
    due to the application of a mask.

    For an n-point function using SHTs, this is the ratio of 
    area weighted by the nth power of the mask to the full sky area 4 pi.
    This simplifies to mean(mask**n) for equal area pixelizations like
    healpix. For SHTs on CAR, it is sum(mask**n * pixel_area_map) / 4pi.
    When using FFTs, it is the area weighted by the nth power normalized
    to the area of the map. This also simplifies to mean(mask**n)
    for equal area pixels. For CAR, it is sum(mask**n * pixel_area_map) 
    / sum(pixel_area_map).

    If not, it does an expensive calculation of the map of pixel areas. If this has
    been pre-calculated, it can be provided as the pmap argument.
    
    """
    assert mask.ndim==1 or mask.ndim==2
    if pmap is None: 
        if equal_area:
            npix = mask.size
            pmap = 4*np.pi / npix if sht else enmap.area(mask.shape,mask.wcs) / npix
        else:
            pmap = enmap.pixsizemap(mask.shape,mask.wcs)
    return np.sum((mask**n)*pmap) /np.pi / 4. if sht else np.sum((mask**n)*pmap) / np.sum(pmap)


def get_camb_spectra(lmax=512, tensor=True, ns=0.965, As=2e-9, r=0.1):
    """
    Returns CMB Cls [TT, EE, BB, TE] from CAMB with low accuracy for fast testing.
    This function is included for completeness to show how the power spectra used
    in this test were generated.

    >> ps = get_camb_spectra(lmax=lmax, r=0.05)  # r sets BB amplitude
    >> ells = np.arange(ps.shape[-1])
    >> np.savez_compressed("pywiggle/data/test_camb_cl.npz", ps=ps, ells=ells)
    
    """
    import camb
    from camb import model
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = tensor
    pars.AccuracyBoost = 0.1
    pars.lAccuracyBoost = 0.1
    pars.HighAccuracyDefault = False

    results = camb.get_results(pars)
    cls = results.get_total_cls(lmax=lmax,CMB_unit='muK')

    # cls has shape (lmax+1, 4): TT, EE, BB, TE
    ps = np.zeros((3, 3, lmax+1))
    ps[0, 0] = cls[:, 0]  # TT
    ps[1, 1] = cls[:, 1]  # EE
    ps[2, 2] = cls[:, 2]  # BB
    ps[0, 1] = ps[1, 0] = cls[:, 3]  # TE

    return ps


def load_test_spectra():
    """
    Load CAMB test Cls from bundled .npz file using pkgutil (compatible with editable installs).
    """
    data = pkgutil.get_data("pywiggle", "data/test_camb_cl.npz")
    if data is None:
        raise FileNotFoundError("Could not load 'data/test_camb_cl.npz' from package.")
    with io.BytesIO(data) as f:
        npz = np.load(f)
        return npz["ps"], npz["ells"]


def cosine_apodize(bmask,width_deg):
    r = width_deg * np.pi / 180.
    return 0.5*(1-np.cos(bmask.distance_transform(rmax=r)*(np.pi/r)))

def test_recover_tensor_Bmode():
    # Sim config ---

    # from orphics import maps
    # mask = maps.cosine_apodize(enmap.downgrade(enmap.read_map('/data5/act/masks/dr6v4_lensing_20250318_masks/baseline/dr6v4_lensing_20251803_night_enhanced_mask_70.fits'),16), 3.)
    # shape,wcs = mask.shape,mask.wcs
    
    res = 8.0 / 60. # deg
    nside = 512
    shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res))
    lmax = 3*nside
    area_deg2 = 4000.
    apod_deg = 10.0
    radius_deg = np.sqrt(area_deg2 / np.pi)
    radius_rad = np.deg2rad(radius_deg)
    np.random.seed(10)

    # Load CMB Cls ---
    ps, ells = load_test_spectra()
    assert ps.shape == (3, 3, len(ells))

    # Simulate polarization map ---
    alm = cs.rand_alm(ps, lmax=lmax)
    bb_orig = cs.alm2cl(alm[2])
    ells = np.arange(bb_orig.size)
    ee_orig = cs.alm2cl(alm[1])
    polmap = cs.alm2map(alm[1:], enmap.empty((2,)+shape, wcs,dtype=np.float32), spin=2)  # only Q,U
    hmap = hp.alm2map(alm,nside,pol=True)
    Qh = hmap[1]
    Uh = hmap[2]
    Q = polmap[0].copy()
    U = polmap[1].copy()

    # Make apodized circular mask
    modrmap = enmap.modrmap(shape,wcs)
    bmask = enmap.zeros(shape,wcs)
    bmask[modrmap<radius_rad] = 1
    dist      = enmap.distance_transform(bmask == 1)
    mask = np.clip(dist / np.deg2rad(apod_deg), 0, 1)
    mask = 0.5 - 0.5*np.cos(np.pi * mask)   # raised-cosine window


    maskh = reproject.map2healpix(mask, nside=nside , method="spline", order=1, extensive=False)

    # Mode decoupling
    mask_alm = cs.map2alm(mask, lmax=2 * lmax)

    # Binning
    b = nmt.NmtBin.from_nside_linear(nside, 16)
    leff = b.get_effective_ells()
    nbins = leff.size
    bin_edges = []
    for i in range(nbins):
        bin_edges.append(b.get_ell_min(i))
    bin_edges.append(b.get_ell_max(nbins-1))

    # bin_edges = np.append([2,10,20], np.arange(40,lmax,10))
    bcents = leff #(bin_edges[1:]+bin_edges[:-1])/2.
    

    masked = polmap*mask
    from orphics import io as oio
    # oio.hplot(masked,'masked',grid=True,colorbar=True,downgrade=4,ticks=30,mask = 0)
    # oio.hplot(mask,'mask',grid=True,colorbar=True,downgrade=4,ticks=30)

    w2 = wfactor(2,mask)
    oalm = cs.map2alm(masked,lmax=lmax,spin=2)
    bb_masked = cs.alm2cl(oalm[1],oalm[1])
    bb_masked = bb_masked / w2
    
    ee_masked = cs.alm2cl(oalm[0],oalm[0])
    ee_masked = ee_masked / w2

    # Run purification ---

    def compute_master(f_a, f_b, wsp):
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        return cl_decoupled

    with bench.show("pure namaster"):
        f2yp = nmt.NmtField(maskh, [Qh, Uh], purify_e=False, purify_b=True,n_iter=0)
        w_yp = nmt.NmtWorkspace.from_fields(f2yp, f2yp, b)
        cl_yp_nmt = compute_master(f2yp, f2yp, w_yp)
        cl_p_bb = cl_yp_nmt[3]

        
        f2np = nmt.NmtField(maskh, [Qh, Uh], purify_e=False, purify_b=False,n_iter=0)
        w_np = nmt.NmtWorkspace.from_fields(f2np, f2np, b)
        cl_np_nmt = compute_master(f2np, f2np, w_np)
        cl_np_ee = cl_np_nmt[2]
        

    
    pureE, pureB = pywiggle.get_pure_EB_alms(Q, U, mask,lmax=lmax)
    # pureE2, pureB2 = pywiggle.get_pure_EB_alms(Q*mask, U*mask, mask,masked_on_input=True,lmax=lmax)

    ialms = np.zeros((2,oalm[0].size),dtype=np.complex128)
    ialms[0]  = oalm[0]
    ialms[1] = pureB # impure E, pure B
    w = pywiggle.Wiggle(lmax, bin_edges=bin_edges)
    w.add_mask('m', mask_alm)
    Mm_np = w.get_coupling_matrix_from_ids('m','m','--',npure=0)
    Mm_p = w.get_coupling_matrix_from_ids('m','m','--',npure=1)

    oio.plot_img(Mm_np,'impureMm.png')
    oio.plot_img(Mm_p,'pureMm.png')

    Mp_np = w.get_coupling_matrix_from_ids('m','m','++',npure=0)
    Mp_p = w.get_coupling_matrix_from_ids('m','m','++',npure=1)

    oio.plot_img(Mp_np,'impureMp.png')
    oio.plot_img(Mp_p,'pureMp.png')
    
    ret = w.decoupled_cl(ialms,ialms, 'm',return_theory_filter=False,pure_B = True)
    
    cl_EE = ret['EE']['Cls']
    cl_BB = ret['BB']['Cls']
    
    ialms[0]  = oalm[0]
    ialms[1] = oalm[1] # impure E, impure B
    w = pywiggle.Wiggle(lmax, bin_edges=bin_edges)
    w.add_mask('m', mask_alm)
    ret = w.decoupled_cl(ialms,ialms, 'm',return_theory_filter=False,pure_B = False)
    icl_EE = ret['EE']['Cls']
    icl_BB = ret['BB']['Cls']

    # Compute power spectrum and compare ---
    bpow = cs.alm2cl(pureB) / w2
    epow = cs.alm2cl(pureE) / w2
    # obpow = cs.alm2cl(pureB2) / w2
    # oepow = cs.alm2cl(pureE2) / w2
    input_bb = ps[2, 2]
    input_ee = ps[1, 1]
    ls = np.arange(input_bb.size)
    ell = np.arange(bpow.size)
    # oell = np.arange(obpow.size)
    plt.figure()
    ell = np.arange(len(bpow))
    plt.plot(ls, input_bb, label='Input BB',ls='--')
    plt.plot(ells, bb_orig, label='Full-sky unmasked BB power',alpha=0.5)
    plt.plot(ells, bb_masked, label='Masked BB power divided by mean(mask**2)')
    plt.plot(ell, bpow, label='Recovered pure B')
    plt.plot(bcents,cl_BB, label = 'Decoupled pure B', marker='d', ls='none')
    plt.plot(bcents,icl_BB, label = 'Decoupled impure B', marker='o', ls='none')
    plt.plot(leff,cl_p_bb, label = 'Decoupled pure B (Nmt)', marker='x', ls='none')
    
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

    plt.figure()
    ell = np.arange(len(epow))
    plt.plot(ls, input_ee, label='Input EE',ls='--')
    plt.plot(ells, ee_orig, label='Full-sky unmasked EE power',alpha=0.5)
    plt.plot(ells, ee_masked, label='Masked EE power / mean(mask**2)')
    plt.plot(ell, epow, label='Recovered pure E')
    # plt.plot(oell, oepow, label='Recovered pure E (masked on input)')
    plt.plot(bcents,icl_EE, label = 'Decoupled impure E', marker='o', ls='none')
    plt.plot(bcents+1,cl_EE, label = 'Decoupled impure E, purified B', marker='o', ls='none')
    plt.plot(leff,cl_np_ee, label = 'Decoupled impure E (Nmt)', marker='x', ls='none')
    plt.xlim(2, 300)
    plt.yscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell^{EE}$')
    plt.legend()
    plt.title(f'E-mode recovery test ({area_deg2:.0f} deg$^2$ mask)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('emodes.png',dpi=200)
