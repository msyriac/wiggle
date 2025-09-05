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



s = stats.Statistics.load_reduced('stats.npz')

bth = io.load_dict('bth.h5')
bth_pure = io.load_dict('bth_pure.h5')
theory = io.load_dict('theory.h5')
bin_edges = np.loadtxt('bin_edges.txt')

utils.analyze(s,bth,bth_pure,theory,bin_edges)
print("Done.")
