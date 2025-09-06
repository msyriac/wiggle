import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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


lmaxs = [1024, 4096]
spins = [0,2]
codes = {
    'wiggle':['$\\mathtt{wiggle}$','k','o'],
    'ducc':['$\\mathtt{ducc}$','b','^'],
    'nmt':['$\\mathtt{NaMaster}$','g','d'],
    'pspy':['$\\mathtt{pspy}$','r','s']
}

for lmax in lmaxs:
    for spin in spins:

        figsize=(6, 5)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        
        d = {}
        for code in codes.keys():
            d[code] = {}
            for b in ['binned','unbinned']:
                d[code][b] = {}
                bstr = '_binned' if b=='binned' else ''
                try:
                    nt,time = np.loadtxt(f'bench_{code}_lmax_{lmax}_spin_{spin}{bstr}.csv',unpack=True,skiprows=1,delimiter=',')
                except:
                    continue
        
                ax.plot(nt, 1.0 / time,
                         marker=codes[code][2], linewidth=2, markersize=6,color=codes[code][1],
                         label=codes[code][0] if b=='binned' else None,ls='--' if b=='binned' else '-'
                         )

        ax.set_xlabel("Number of OpenMP Threads", fontsize=12)
        ax.set_ylabel("Average Execution Speed ($s^{-1}$)", fontsize=12)
        ax.set_yscale('log')
        ax.set_title(f"$l_{{\\rm max}}={lmax}$; spin={spin}", fontsize=14)
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        # ax.axhline(y=0, ls='--', color='k')
        ax.legend(loc='best')
        ax.set_xticks(nt)
        plt.savefig(f'plot_{lmax}_{spin}.png',bbox_inches='tight')
