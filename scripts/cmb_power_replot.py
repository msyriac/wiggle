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

from orphics import io, stats, maps
import utils

# snside = '' 
# nside = 2048

# s = stats.Statistics.load_reduced(f'stats{snside}.npz')

# bth = io.load_dict(f'bth{snside}.h5')
# bth_pure = io.load_dict(f'bth_pure{snside}.h5')
# theory = io.load_dict(f'theory{snside}.h5')
# bin_edges = np.loadtxt(f'bin_edges{snside}.txt')

# utils.analyze(s,bth,bth_pure,theory,bin_edges,nside,do_pure=False)

# snside = '_256' 
# nside = 256

# s = stats.Statistics.load_reduced(f'stats{snside}.npz')

# bth = io.load_dict(f'bth{snside}.h5')
# bth_pure = io.load_dict(f'bth_pure{snside}.h5')
# theory = io.load_dict(f'theory{snside}.h5')
# bin_edges = np.loadtxt(f'bin_edges{snside}.txt')

# utils.analyze(s,bth,bth_pure,theory,bin_edges,nside,do_pure=True)



snside = '_512' 
nside = 512

s = stats.Statistics.load_reduced(f'stats{snside}.npz')

bth = io.load_dict(f'bth{snside}.h5')
bth_pure = io.load_dict(f'bth_pure{snside}.h5')
theory = io.load_dict(f'theory{snside}.h5')
bin_edges = np.loadtxt(f'bin_edges{snside}.txt')

utils.analyze(s,bth,bth_pure,theory,bin_edges,nside,do_pure=True)



print("Done.")
