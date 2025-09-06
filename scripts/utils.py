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
        ell_min = 2, ell_max = 3000,
        Dl_min = 1e-3, Dl_max = 1e4,  # μK^2
        figsize = (9.0, 5.0),
        title = None, plot_name='plot.png',xscale="log",pure=True,rlim=0.2,ylim=None,
        plegloc = 'center left',tlegloc = 'center left'
        
):
    # Colors & labels
    spec_meta = {
        "TT": {"color": "blue", "label": "TT", 'marker':'o'},
        "EE": {"color": "green", "label": "EE", 'marker':'s'},
        "BB": {"color": "red", "label": "BB", 'marker':'^'},
        "BB pure": {"color": "orange", "label": "BB pure", 'marker':'D','ls':'-'},
        "TE": {"color": "purple", "label": "TE", 'marker':'8'},
        "TB": {"color": "cyan", "label": "TB", 'marker':'d'},
        "EB": {"color": "magenta", "label": "EB", 'marker':'*'},
        "TB pure": {"color": "pink", "label": "TB pure", 'marker':'d','ls':':'},
        "EB pure": {"color": "maroon", "label": "EB pure", 'marker':'d','ls':'--'},
    }
    

    # Build figure with skinny residuals panel
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(nrows=4 if pure else 3, ncols=1, height_ratios=[3.2, 2.3, 1.3, 1.3] if pure else [3.2, 2.3, 1.3] , hspace=0.04)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1], sharex=ax)
    axr = fig.add_subplot(gs[2], sharex=ax)
    if pure:
        axb = fig.add_subplot(gs[3], sharex=ax)

    # MAIN PANEL: log-log theory + data
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.set_xlim(ell_min, ell_max)
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])

    # Grid tuned for log axes
    ax.grid(True, which="both", ls="-", lw=0.4, alpha=0.25)
    ax.set_ylabel(r"$D_l [\mu\mathrm{K}^2]$")

    # Tick behavior: minor labels off for x-log to reduce clutter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10) * 0.1))

    # Plot theory curves
    for spec in ["TT", "EE", "BB","TE"]:
        Cl = theory[spec]
        ells = np.arange(Cl.size)
        m = np.isfinite(ells) & np.isfinite(Cl) & (ells > 0) 
        if not np.any(m):
            continue
        Dl = Cl[m] * (ells[m]*(ells[m]+1.))/2./np.pi
        if spec=='TE':
            oax = axt
        else:
            oax = ax
        oax.plot(ells[m], Dl,
                lw=1.2, alpha=0.9, color=spec_meta[spec]["color"])

    # Plot binned data with error bars
    markersize=5
    dspecs1 = ['TT','EE','BB', 'BB pure']
    for spec in dspecs1:
        if not(pure) and 'pure' in spec: continue
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
                    fmt=spec_meta[spec]["marker"], ms=markersize, mec="none",
                    alpha=0.95, color=spec_meta[spec]["color"],
                    label=f"{spec}")

    if title:
        ax.set_title(title, pad=6)

    # LEGEND
    # bbox_to_anchor=(1, 0.5)
    # ax.legend(ncols=2, frameon=1, loc="center left", handlelength=1.6,numpoints=1,bbox_to_anchor=bbox_to_anchor)
    ax.legend(ncols=2, frameon=1, loc=plegloc, handlelength=1.6,numpoints=1)


    axt.set_xscale(xscale)
    axt.set_xlim(ell_min, ell_max)
    axt.set_ylabel(r"$D_l [\mu\mathrm{K}^2]$")
    axt.set_xlabel(r"Multipole $l$")
    axt.grid(True, which="both", ls="-", lw=0.4, alpha=0.25)
    axt.axhline(1.0, color="k", lw=1.0, alpha=0.8, ls='--')

    dspecs2 = ['TE','TB','EB', 'TB pure', 'EB pure']
    for spec in dspecs2:
        if not(pure) and 'pure' in spec: continue
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
        axt.errorbar(ell[m], Dl[m], yerr=yerr_use,
                    fmt=spec_meta[spec]["marker"], ms=markersize, mec="none",
                    alpha=0.95, color=spec_meta[spec]["color"],
                    label=f"{spec}")

    axt.legend(ncols=2, frameon=1, loc=tlegloc, handlelength=1.6,numpoints=1)
    
    
    # RESIDUALS PANEL: (data - theory)/theory, semilogx (log x, linear y)
    axr.axhline(0.0, color="k", lw=1.0, alpha=0.8, ls='--')
    axr.set_xscale(xscale)
    axr.set_xlim(ell_min, ell_max)
    axr.set_ylabel(r"$\Delta C_{l}/C_{l}^{\rm theory}$")
    axr.set_xlabel(r"Multipole $l$")
    axr.grid(True, which="both", ls="-", lw=0.4, alpha=0.25)

    # Small y-range centered on 0 for clarity; adjust as needed
    axr.set_ylim(-rlim, rlim)
    axr.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # Compute & plot residuals for each spectrum where theory exists
    dspecs = dspecs1 + dspecs2
    nspec = len(dspecs)
    i = 0
    for spec in dspecs:
        if not(pure) and 'pure' in spec: continue
        th = binned_th.get(spec, {})
        d = data.get(spec, {})
        ell_d, Cl_d, yerr_d = d.get("ell"), d.get("Cl"), d.get("yerr")
        m = np.isfinite(ell_d)  & (ell_d > 0)

        Cl_th = th["Cl"][m]
        # Avoid division by zero
        good = Cl_th > 0
        # if not np.any(good):
        #     continue

        resid = (Cl_d[m][good] - Cl_th[good]) / Cl_th[good]
        yerr_frac = None
        if yerr_d is not None:
            yerr_frac = np.array(yerr_d)[m][good] / Cl_th[good]
            
        #off = np.linspace(-5,5,nspec)[i]
        off = 0.
        axr.errorbar(ell_d[m][good]+off, resid, yerr=yerr_frac,
                     fmt=spec_meta[spec]["marker"], ms=markersize, mec="none",
                     alpha=0.95, color=spec_meta[spec]["color"])
        i = i + 1

    if pure:
        axb.axhline(1.0, color="k", lw=1.0, alpha=0.8, ls='--')
        for s in ['BB','EB','TB']:
            axb.plot(data[f'{s} pure']['ell'], data[f'{s} pure']['yerr']/data[s]['yerr'],
                lw=1.2, alpha=0.9, color=spec_meta[f'{s} pure']["color"],
                     label=f"{s} pure uncertainty ratio",marker=spec_meta[f'{s} pure']["marker"],ms=markersize,ls=spec_meta[f'{s} pure']["ls"])

        axb.set_xscale(xscale)
        axb.set_xlim(ell_min, ell_max)
        axb.set_ylabel(r"$\sigma(C_{l}^{\rm pure})/\sigma(C_{l})$")
        axb.set_xlabel(r"Multipole $l$")
        axb.grid(True, which="both", ls="-", lw=0.4, alpha=0.25)
        # axb.legend(ncols=2, frameon=1, loc=legloc, handlelength=1.6,numpoints=1)
        

    # Tight layout with shared x
    plt.setp(ax.get_xticklabels(), visible=False)
    fig.align_ylabels([ax, axr, axb]) if pure else fig.align_ylabels([ax, axr])
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.12, hspace=0.02)
    plt.savefig(plot_name,bbox_inches='tight')
    plt.close()


def analyze(s,bth,bth_pure,theory,bin_edges,nside,do_pure=False):
    cents = (bin_edges[1:] + bin_edges[:-1])/2.

    data = {}
    binned_th = {}
    #['TT','EE','TE','BB','EB', 'TB','BB pure']
    for spec in ['TT','EE','BB','BB pure','TE','TB pure', 'TB', 'EB', 'EB pure']:
        if not(do_pure) and 'pure' in spec: continue
        y = s.mean(spec)
        yerr = np.sqrt(s.var(spec))
        data[spec] = {}
        data[spec]['ell'] = cents
        data[spec]['Cl'] = y.copy()
        data[spec]['yerr'] = yerr.copy()/np.sqrt(s.count(spec))

        binned_th[spec] = {}
        binned_th[spec]['ell'] = cents
        if 'pure' in spec:
            binned_th[spec]['Cl'] = bth_pure[spec[:2]].copy()
        else:
            binned_th[spec]['Cl'] = bth[spec].copy()





    if do_pure:
        plot_cmb_spectra_with_residuals(
            theory,
            data,
            binned_th,ell_min=2,ell_max=300,xscale='log',
            plot_name=f'plow_{nside}.png',figsize = (5.0, 7.0),plegloc='center left',tlegloc='lower left')

    else:
        plot_cmb_spectra_with_residuals(
            theory,
            data,
            binned_th,ell_min=300,ell_max=4096,xscale='linear',plot_name=f'phigh_{nside}.png',
            figsize = (7.0, 5.0),pure=False,rlim=0.05,ylim=[2e-3,1e4],plegloc='upper right',tlegloc='lower right')
        
