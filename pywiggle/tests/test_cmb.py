import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from pixell import enmap, enplot, curvedsky as cs, utils, bench,reproject
import matplotlib.pyplot as plt
from pywiggle import utils as wutils
import pywiggle
import io,sys
import healpy as hp
from collections import defaultdict
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from orphics import io, stats # !!!

np.random.seed(100)

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "mathtext.rm": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "errorbar.capsize": 2.5,
})



def plot_cmb_spectra_with_residuals(
        theory,
        data,
        binned_th,
        ell_min = 15, ell_max = 3000,
        Dl_min = 1e-3, Dl_max = 1e4,  # μK^2
        figsize = (6.0, 5.0),
        title = None, plot_name='plot.png',xscale="log"
):
    # Colors & labels
    spec_meta = {
        "TT": {"color": "#1f77b4", "label": "TT"},
        "EE": {"color": "#2ca02c", "label": "EE"},
        "BB": {"color": "#d62728", "label": "BB"},
        "BB pure": {"color": "orange", "label": "BB pure"},
    }
    

    # Build figure with skinny residuals panel
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[3.2, 1.3, 1.3], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    axb = fig.add_subplot(gs[2], sharex=ax)

    # MAIN PANEL: log-log theory + data
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.set_xlim(ell_min, ell_max)

    # Grid tuned for log axes
    ax.grid(True, which="both", ls="-", lw=0.4, alpha=0.25)
    ax.set_ylabel(r"$D_l [\mu\mathrm{K}^2]$")

    # Tick behavior: minor labels off for x-log to reduce clutter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10) * 0.1))

    # Plot theory curves
    for spec in ["TT", "EE", "BB"]:
        Cl = theory[spec]
        ells = np.arange(Cl.size)
        m = np.isfinite(ells) & np.isfinite(Cl) & (ells > 0) 
        if not np.any(m):
            continue
        Dl = Cl[m] * (ells[m]*(ells[m]+1.))/2./np.pi
        ax.plot(ells[m], Dl,
                lw=1.8, alpha=0.9, color=spec_meta[spec]["color"])

    # Plot binned data with error bars
    markersize=5
    dspecs = data.keys()
    for spec, marker in zip(dspecs, ["o", "s", "^","d"]):
        d = data.get(spec, {})
        ell, Cl, cyerr = d.get("ell"), d.get("Cl"), d.get("yerr")
        Dl = (ell*(ell+1.))*Cl/2./np.pi
        yerr = (ell*(ell+1.))*cyerr/2./np.pi
        if ell is None or Dl is None:
            continue
        m = np.isfinite(ell) & np.isfinite(Dl) & (ell > 0) 
        if not np.any(m):
            continue
        yerr_use = None
        if yerr is not None:
            yerr_use = np.array(yerr)[m]
        ax.errorbar(ell[m], Dl[m], yerr=yerr_use,
                    fmt=marker, ms=markersize, mec="none",
                    alpha=0.95, color=spec_meta[spec]["color"],
                    label=f"{spec}")

    if title:
        ax.set_title(title, pad=6)

    # LEGEND
    # bbox_to_anchor=(1, 0.5)
    # ax.legend(ncols=2, frameon=1, loc="center left", handlelength=1.6,numpoints=1,bbox_to_anchor=bbox_to_anchor)
    ax.legend(ncols=2, frameon=1, loc="center left", handlelength=1.6,numpoints=1)


    # RESIDUALS PANEL: (data - theory)/theory, semilogx (log x, linear y)
    axr.axhline(0.0, color="k", lw=1.0, alpha=0.8, ls='--')
    axr.set_xscale(xscale)
    axr.set_xlim(ell_min, ell_max)
    axr.set_ylabel(r"$\Delta C_{l}/C_{l}$")
    axr.set_xlabel(r"Multipole $l$")
    axr.grid(True, which="both", ls="-", lw=0.4, alpha=0.25)

    # Small y-range centered on 0 for clarity; adjust as needed
    axr.set_ylim(-0.2, 0.2)
    axr.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # Compute & plot residuals for each spectrum where theory exists
    nspec = len(dspecs)
    i = 0
    for spec, marker in zip(dspecs, ["o", "s", "^","d"]):
        th = binned_th.get(spec, {})
        d = data.get(spec, {})
        if th.get("ell") is None or th.get("Cl") is None:
            continue
        ell_d, Cl_d, yerr_d = d.get("ell"), d.get("Cl"), d.get("yerr")
        if ell_d is None or Cl_d is None:
            continue
        m = np.isfinite(ell_d)  & (ell_d > 0)
        if not np.any(m):
            continue

        Cl_th = th["Cl"][m]
        # Avoid division by zero
        good = Cl_th > 0
        if not np.any(good):
            continue

        resid = (Cl_d[m][good] - Cl_th[good]) / Cl_th[good]
        yerr_frac = None
        if yerr_d is not None:
            yerr_frac = np.array(yerr_d)[m][good] / Cl_th[good]
            
        #off = np.linspace(-5,5,nspec)[i]
        off = 0.
        axr.errorbar(ell_d[m][good]+off, resid, yerr=yerr_frac,
                     fmt=marker, ms=markersize, mec="none",
                     alpha=0.95, color=spec_meta[spec]["color"])
        i = i + 1

    axb.axhline(1.0, color="k", lw=1.0, alpha=0.8, ls='--')
    axb.plot(data['BB pure']['ell'], data['BB pure']['yerr']/data['BB']['yerr'],
            lw=1.8, alpha=0.9, color=spec_meta['BB pure']["color"],
             label=f"BB pure uncertainty ratio",marker='d',ms=markersize)
    
    axb.set_xscale(xscale)
    axb.set_xlim(ell_min, ell_max)
    axb.set_ylabel(r"$\sigma(C_{l}^{\rm pure})/\sigma(C_{l})$")
    axb.set_xlabel(r"Multipole $l$")
    axb.grid(True, which="both", ls="-", lw=0.4, alpha=0.25)
        

    # Tight layout with shared x
    plt.setp(ax.get_xticklabels(), visible=False)
    fig.align_ylabels([ax, axr, axb])
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.12, hspace=0.02)
    plt.savefig(plot_name,bbox_inches='tight')
    plt.close()



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

bin_edges = np.append(np.append(np.geomspace(2,200,10),np.arange(240,2000,40)),np.arange(2250,8000,250))
bin_edges = bin_edges[bin_edges<lmax]
cents = (bin_edges[1:] + bin_edges[:-1])/2.



with bench.show("camb"):
    ps = wutils.get_camb_spectra(lmax=tlmax)
ells = np.arange(ps[0,0].size)
assert ps.shape == (3, 3, len(ells))

nsims = 1


def unpack_cls(ret):
    include_keys = ['TT','EE','TE','EB','BE','BB','TB']
    return {k: v['Cls'] for k, v in ret.items() if k in include_keys}


theory = wutils.get_cldict(ps)

s = stats.Statistics()

for i in range(nsims):
    print(i)
    with bench.show("sim"):
        alm = cs.rand_alm(ps, lmax=tlmax)

        imap = cs.alm2map(alm, enmap.empty((3,)+shape, wcs,dtype=np.float32))
    if i==0:
        with bench.show("mask"):
            try:
                mask = enmap.read_map(f'mask_{nside}.fits')
                mask_alm = hp.read_alm(f'mask_alm_{nside}.fits')
            except:
                _,mask = wutils.get_mask(nside,shape,wcs,radius_deg,apod_deg,smooth_deg=smooth_deg)
                mask_alm = cs.map2alm(mask,lmax=2*lmax,spin=0)
                enmap.write_map(f'mask_{nside}.fits',mask)
                hp.write_alm(f'mask_alm_{nside}.fits',mask_alm,overwrite=True)
                
        with bench.show("purify init"):
            ebp = pywiggle.EBPurifier(mask,lmax)

    omap = imap * mask
    with bench.show("map2alm"):
        oalms = cs.map2alm(omap,lmax=lmax,spin=[0,2])

    if i==0 and lmax<256:
        for j in range(3): wutils.hplot(imap[j],f'imap_{j}',grid=True,ticks=20)
        wutils.hplot(mask,f'mask',grid=True,ticks=20)

    with bench.show("purify"):
        # Purify B
        _, pureB = ebp.project(omap[1], omap[2],masked_on_input=True)


    # Get impure power
    with bench.show("wiggle"):
        ret = pywiggle.get_powers(oalms,oalms, mask_alm,return_theory_filter=True,lmax=lmax,bin_edges=bin_edges)
    bcls = unpack_cls(ret)
    for spec in ['TT','EE','TE','BB','EB','TB']:
        s.add(spec,bcls[spec])
    
    if i==0:
        bth = pywiggle.get_binned_theory(ret,theory)

    # Get pure power
    oalms[2] = pureB.copy()
    with bench.show("wiggle"):
        ret_pure = pywiggle.get_powers(oalms,oalms, mask_alm,return_theory_filter=True,lmax=lmax,bin_edges=bin_edges,pure_B=True)
    bcls_pure = unpack_cls(ret_pure)
    s.add('BB pure',bcls_pure['BB'])
    if i==0:
        bth_pure = pywiggle.get_binned_theory(ret_pure,theory)

s.allreduce()

data = {}
binned_th = {}
#['TT','EE','TE','BB','EB', 'TB','BB pure']
for spec in ['TT','EE','BB','BB pure']:
    y = s.mean(spec)
    yerr = np.sqrt(s.var(spec))
    data[spec] = {}
    data[spec]['ell'] = cents
    data[spec]['Cl'] = y.copy()
    data[spec]['yerr'] = yerr.copy()/np.sqrt(nsims)

    binned_th[spec] = {}
    binned_th[spec]['ell'] = cents
    if spec=='BB pure':
        binned_th[spec]['Cl'] = bth_pure['BB'].copy()
    else:
        binned_th[spec]['Cl'] = bth[spec].copy()

    


plot_cmb_spectra_with_residuals(
    theory,
    data,
    binned_th,
    ell_max=lmax)
        
