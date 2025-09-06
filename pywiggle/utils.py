import numpy as np
from . import _wiggle
import os
from time import time
import pkgutil, io
import healpy as hp
import pywiggle

# For benchmarking and testing
try:
    import ducc0
    from pspy._mcm_fortran import mcm_compute as mcm_fortran
    import pymaster as nmt
except:
    pass


"""
=========
Utilities
=========
"""

def change_alm_lmax(alms, lmax, mmax_out=None):
    ilmax  = hp.Alm.getlmax(alms.shape[-1], mmax_out)
    immax  = mmax_out or ilmax          # fall back to full m-range
    ommax  = mmax_out or lmax
    return np.asarray(hp.sphtfunc.resize_alm(alms, ilmax, immax, lmax, ommax))

def multipole_to_bin_indices(lmax, bin_edges):
    """
    Returns an array mapping multipole index l (0 to lmax) to bin index,
    based on provided bin_edges.

    Parameters:
    - lmax: int, maximum multipole
    - bin_edges: 1D array-like, sorted bin edges of length nbins + 1

    Returns:
    - bin_indices: np.ndarray of shape (lmax + 1,), where bin_indices[l] gives the bin index for multipole l
    """
    bin_edges = np.asarray(bin_edges)
    if bin_edges.min()<0: raise ValueError
    if bin_edges.max()>(lmax+1): raise ValueError
    if np.any(np.diff(bin_edges)<=0): raise ValueError("Bin edges must be monotonically increasing.")

    ls = np.arange(lmax + 1)
    bin_indices = np.searchsorted(bin_edges, ls, side='right') - 1
    return bin_indices


def bin_array(arr, bin_indices, nbins, weights=None):
    """
    Bin an array using bin indices and optional per-element weights.

    Parameters
    ----------
    arr : ndarray of shape (lmax + 1,)
        Array defined over multipole ℓ values.
    bin_indices : ndarray of shape (lmax + 1,)
        Per-ℓ bin index (-1 for entries to ignore).
    weights : ndarray of shape (lmax + 1,), optional
        Optional per-element weights w_ell. If not provided, defaults to 1.

    Returns
    -------
    binned : ndarray of shape (nbins,)
        Array with weighted sum of values in each bin.
    """
    arr = np.asarray(arr)
    bin_indices = np.asarray(bin_indices)
    
    if arr.shape != bin_indices.shape:
        raise ValueError("arr and bin_indices must have the same shape")
    
    if weights is None:
        weights = np.ones_like(arr, dtype=np.float64)
    else:
        weights = np.asarray(weights)
        if weights.shape != arr.shape:
            raise ValueError("weights must have the same shape as arr")

    # Mask out entries with bin_index == -1 or >= nbins
    mask = np.logical_and(bin_indices >= 0, bin_indices < nbins)
    valid_bins = bin_indices[mask]
    valid_vals = arr[mask]
    valid_weights = weights[mask]

    weighted_vals = valid_vals * valid_weights

    # Vectorized binning with weights
    binned = np.bincount(valid_bins, weights=weighted_vals, minlength=nbins)

    return binned

def normalize_weights_per_bin(nbins, bin_indices, weights):
    """
    Normalize weights so that values in each bin sum to 1.

    Parameters
    ----------
    nbins : int
        Number of bins.
    bin_indices : ndarray of shape (N,)
        Bin index for each element. Use -1 for elements to be ignored.
    weights : ndarray of shape (N,)
        Weights to normalize.

    Returns
    -------
    normalized_weights : ndarray of shape (N,)
        Weights normalized per bin.
    """
    bin_indices = np.asarray(bin_indices)
    weights = np.asarray(weights)

    # Sum weights per bin (bins with no entries will have sum=0)
    bin_sums = np.bincount(bin_indices[bin_indices >= 0], weights[bin_indices >= 0], minlength=nbins)

    # Get the sum for each entry’s bin
    per_point_sums = bin_sums[bin_indices.clip(0)]

    # Avoid division by zero for bins with no data
    norm_factors = np.where(per_point_sums > 0, per_point_sums, 1.0)

    # Normalize
    normalized = np.where(bin_indices >= 0, weights / norm_factors, 0.0)
    return normalized

def get_wigner_d(lmax,s1,s2,cos_thetas,
                 bin_edges=None,bin_weights=None,bin_weights2=None):
    r"""
    Compute Wigner-d matrices for given spins and angles, with optional weighted binning over multipoles.

    This function evaluates the Wigner small-d matrices, \( d^{\ell}_{s_1,s_2}(\theta) \), for 
    multipoles from 0 up to `lmax` at the specified cosines of angles (`cos_thetas`). 
    It supports optional binning of the multipole axis using specified bin edges 
    and weighting schemes.

    Parameters
    ----------
    lmax : int
        Maximum multipole (ℓ) for which to compute the Wigner-d matrices.

    s1 : int
        First spin index for the Wigner-d matrix.

    s2 : int
        Second spin index for the Wigner-d matrix.

    cos_thetas : array_like
        Array of cosines of the angles θ at which to evaluate the Wigner-d matrices.

    bin_edges : array_like, optional
        Bin edges defining multipole bins for aggregating the Wigner-d matrices.
        Bins are interpreted as intervals: low_edge ≤ ℓ < upper_edge.
        If not provided, no binning is performed and the full multipole range is returned.

    bin_weights : array_like, optional
        Multipole weights used for binning along the ℓ axis. If provided, enables weighted binning.
        If `bin_weights2` is not provided, single weighted binning is performed.

    bin_weights2 : array_like, optional
        Optional second set of multipole weights for double-weighted binning.
        Only used if `bin_weights` is also provided. If `bin_weights2` is provided,
        double weighted binning is performed.

    Returns
    -------
    ndarray
        If no binning is requested (`bin_edges` is None), returns a 2D array of shape 
        `(len(cos_thetas), lmax + 1)` containing the Wigner-d matrices evaluated at each angle 
        and multipole.

        If binning is requested, returns a binned (or double-binned) Wigner-d matrix array
        with shape depending on the number of bins and angles.

    Raises
    ------
    ValueError
        If `bin_weights` is provided without `bin_edges`.

    Notes
    -----
    - This function is intended primarily for testing purposes and is not used in `wiggle`.
    """
    
    if bin_edges is None:
        if bin_weights is not None: raise ValueError
        return _wiggle._compute_wigner_d_matrix(lmax,s1,s2,cos_thetas)
    else:
        nbins, bin_indices, nweights = _prepare_bins(bin_edges,bin_weights,lmax)
        if bin_weights2 is not None:
            _, _, nweights2 = _prepare_bins(bin_edges,bin_weights2,lmax)
            return _wiggle._compute_double_binned_wigner_d(lmax,s1,s2,cos_thetas,nbins,bin_indices,nweights,nweights2)
        else:
            return _wiggle._compute_binned_wigner_d(lmax,s1,s2,cos_thetas,nbins,bin_indices,nweights)


def bin_square_matrix(matrix,bin_edges,lmax,bin_weights=None,bin_weights2=None):
    """
    Bin a square matrix in along both directions using specified multipole binning and weights.

    This function reduces a 2D square matrix (e.g., a coupling matrix)
    from (lmax+1) × (lmax+1) to a binned form of shape (n_bins × n_bins), where binning 
    is defined by multipole `bin_edges` and optional weights. It supports different 
    weighting schemes along each axis of the matrix.

    Parameters
    ----------
    matrix : 2D array_like
        Square matrix of shape (lmax+1, lmax+1) to be binned. Typically a function of 
        multipole indices (ℓ, ℓ').

    bin_edges : array_like
        Array defining the edges of the multipole bins. Must span the relevant ℓ-range 
        up to `lmax`. Note, these are of the form low_edge <= ℓ < upper_edge.

    lmax : int
        Maximum multipole to consider. Defines the size of the unbinned matrix as 
        (lmax+1) × (lmax+1).

    bin_weights : array_like or None, optional
        Weights to apply within each bin along the first axis (rows). If `None`, 
        unit weights are used and internally normalized.

    bin_weights2 : array_like or None, optional
        Weights to apply within each bin along the second axis (columns). If `None`, 
        unit weights are used but not normalized—useful for asymmetric binning 
        or theoretical filters.

    Returns
    -------
    binned_matrix : ndarray
        Binned square matrix of shape (n_bins, n_bins), where `n_bins = len(bin_edges) - 1`.

    Notes
    -----
    - Internally uses normalized weights along the first axis, and unnormalized unity for the second axis if weights for the second axis are not provided. This correctly transforms an unbinned coupling matrix to a binned coupling matrix.
    """
    if bin_weights is None: bin_weights = np.ones(matrix.shape[0]) # meant to be normalized
    if bin_weights2 is None:
        nbins, bin_indices, nweights = _prepare_bins(bin_edges,bin_weights,lmax,bin_weights2=None)
        nweights2 = nweights*0 + 1 # not normalized
    else:
        nbins, bin_indices, nweights, nweights2 = _prepare_bins(bin_edges,bin_weights,lmax,bin_weights2=bin_weights2)
    return _wiggle.bin_matrix(matrix,bin_indices,bin_indices,nweights,nweights2,nbins,nbins)

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
    from pixell import enmap
    assert mask.ndim==1 or mask.ndim==2
    if pmap is None: 
        if equal_area:
            npix = mask.size
            pmap = 4*np.pi / npix if sht else enmap.area(mask.shape,mask.wcs) / npix
        else:
            pmap = enmap.pixsizemap(mask.shape,mask.wcs)
    return np.sum((mask**n)*pmap) /np.pi / 4. if sht else np.sum((mask**n)*pmap) / np.sum(pmap)

def get_camb_spectra(lmax=512, tensor=True, ns=0.965, As=2e-9, r=0.05):
    """
    Returns CMB Cls [TT, EE, BB, TE] from CAMB with low accuracy for fast testing.
    
    """
    import camb
    from camb import model
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = tensor
    pars.AccuracyBoost = 1
    pars.lAccuracyBoost = 1
    pars.HighAccuracyDefault = False

    results = camb.get_results(pars)
    cls = results.get_total_cls(lmax=lmax,CMB_unit='muK',raw_cl=True)

    # cls has shape (lmax+1, 4): TT, EE, BB, TE
    ps = np.zeros((3, 3, lmax+1))
    ps[0, 0] = cls[:, 0]  # TT
    ps[1, 1] = cls[:, 1]  # EE
    ps[2, 2] = cls[:, 2]  # BB
    ps[0, 1] = ps[1, 0] = cls[:, 3]  # TE

    return ps

def get_cldict(ps):
    cl_dict = {}
    cl_dict['TT'] = ps[0,0]
    cl_dict['TE'] = ps[0,1]
    cl_dict['EE'] = ps[1,1]
    cl_dict['BB'] = ps[2,2]
    cl_dict['EB'] = ps[2,2]*0.
    cl_dict['TB'] = ps[2,2]*0.
    return cl_dict

def cosine_apodize(bmask,width_deg):
    r = width_deg * np.pi / 180.
    return 0.5*(1-np.cos(bmask.distance_transform(rmax=r)*(np.pi/r)))

# CONSOLE I/O
def cprint(string,color=None,bold=False,uline=False):
    if not(isinstance(string,str)):
        string = str(string)
    x=""
    if bold:
        x+=bcolors.BOLD
    if uline:
        x+=bcolors.UNDERLINE

    if color is not None: color = color.lower()    
    if color in ['b','blue']:
        x+=bcolors.OKBLUE
    elif color in ['r','red','f','fail']:
        x+=bcolors.FAIL
    elif color in ['g','green','ok']:
        x+=bcolors.OKGREEN
    elif color in ['y','yellow','w','warning']:
        x+=bcolors.WARNING
    elif color in ['p','purple','h','header']:
        x+=bcolors.HEADER
    
    print(x+string+bcolors.ENDC)
    
class bcolors:
    '''
    Colored output for print commands
    '''
    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def hplot(img,savename=None,verbose=True,grid=False,**kwargs):
    from pixell import enplot
    plots = enplot.get_plots(img,grid=grid,**kwargs)
    if savename is None:
        enplot.show(plots)
        return
    enplot.write(savename,plots)
    if verbose: cprint("Saved plot to "+ savename,color="g")

def mollview(hp_map,filename=None,lim=None,coord='C',verbose=True,return_projected_map=False,xsize=1200,grat_deg=None,dpi=None,grat_color='gray',grat_alpha=0.5,**kwargs):
    '''
    mollview plot for healpix wrapper
    '''
    import healpy as hp
    if lim is None:
        cmin = cmax = None
    elif type(lim) is list or type(lim) is tuple:
        cmin,cmax = lim
    else:
        cmin =-lim
        cmax = lim
    retimg = hp.mollview(hp_map,min=cmin,max=cmax,coord=coord,return_projected_map=return_projected_map,xsize=xsize,**kwargs)
    if grat_deg is not None:
        hp.graticule(dpar=grat_deg,dmer=grat_deg,coord=coord,color=grat_color,alpha=grat_alpha)
    if filename is not None:
        plt.savefig(filename,dpi=dpi)
        if verbose: cprint("Saved healpix plot to "+ filename,color="g")
    if return_projected_map: return retimg


"""
=========
Helpers
=========
"""

def _prepare_bins(bin_edges,bin_weights,lmax,bin_weights2=None):
    if (bin_edges is None):
        if bin_weights is not None: raise ValueError
        return None

    bin_indices = multipole_to_bin_indices(lmax, bin_edges)
    
    if (bin_weights is None): bin_weights = np.ones(lmax+1)
    if bin_weights.size<(lmax+1): raise ValueError(f"bin_weights size is {bin_weights.size} but need lmax+1={lmax+1}")
    # TODO: Do other checks like making sure bins are monotonic and non-overlapping.
    bin_weights = bin_weights[:lmax+1]
    nbins = len(bin_edges)-1
    nweights = normalize_weights_per_bin(nbins, bin_indices, bin_weights)
    if bin_weights2 is None:
        return nbins, bin_indices, nweights
    else:
        nweights2 = normalize_weights_per_bin(nbins, bin_indices, bin_weights2)
    return nbins, bin_indices, nweights, nweights2


    
def _parity_flip(c,parity):
    c = c.copy()
    if parity=='-':
        c[1::2] *= -1  # This is (-1)^ell
    elif parity=='+':
        pass
    else:
        raise ValueError
    return c


"""
===================
Benchmark utilities
===================

The following are interfaces to ducc, pspy and Namaster for benchmarking and testing.
"""

# Interface to namaster provided by David Alonso
def _get_mcm_standalone(spin1, spin2, cl_mask, lmax,
                       pureE1=False, pureB1=False,
                       pureE2=False, pureB2=False,
                       beam1=None, beam2=None):
    lmax_mask = len(cl_mask)-1
    if beam1 is None:
        beam1 = np.ones(lmax+1)
    if beam2 is None:
        beam2 = np.ones(lmax+1)
    # (binning is arbitrary, but required by C layer)
    b = nmt.NmtBin.from_lmax_linear(lmax, 10)

    # Call C layer
    wsp = nmt.nmtlib.comp_coupling_matrix(
        int(spin1), int(spin2),
        int(lmax), int(lmax_mask),
        int(pureE1), int(pureB1),
        int(pureE2), int(pureB2),
        0, 0.0, beam1, beam2, cl_mask,
        b.bin, 0, -1, -1, -1)

    # Extract MCM from C layer
    nmap1 = 2 if spin1 else 1
    nmap2 = 2 if spin2 else 1
    ncls = nmap1*nmap2
    nrows = (lmax+1)*ncls
    mcm = nmt.nmtlib.get_mcm(wsp, nrows*nrows).reshape([nrows, nrows])
    nmt.nmtlib.workspace_free(wsp)

    return mcm.reshape([lmax+1, ncls, lmax+1, ncls])

# Copied many of the following from mreinecke/ducc0

def _tri2full(tri, lmax):
    res = np.zeros((tri.shape[0], tri.shape[1], lmax+1, lmax+1))
    lfac = 2.*np.arange(lmax+1) + 1.
    for l1 in range(lmax+1):
        startidx = l1*(lmax+1) - (l1*(l1+1))//2
        res[:,:,l1,l1:] = lfac[l1:] * tri[:,:, startidx+l1:startidx+lmax+1]
        res[:,:,l1:,l1] = (2*l1+1) * tri[:,:, startidx+l1:startidx+lmax+1]
    return res


def _mcm00_nmt(spec,lmax):
    mcm = _get_mcm_standalone(spin1=0, spin2=0, cl_mask=spec[0], lmax=lmax)[:,0,:,0][None,...]
    return mcm
    
def _mcm00_pspy(spec, lmax):
    nspec = spec.shape[0]
    lrange_spec = np.arange(spec.shape[1])
    res=np.zeros((nspec, lmax+1, lmax+1))
    mcmtmp = np.empty((lmax+1, lmax+1))
    for i in range(nspec):
        wcl = spec[i]*(2*lrange_spec+1)
        mcm_fortran.calc_coupling_spin0(wcl, lmax+1, lmax+1, lmax+1, mcmtmp.T)
        mcm_fortran.fill_upper(mcmtmp.T)
        mcmtmp *= (np.arange(2, lmax+3)*2+1.)/(4*np.pi)
        res[i, 2:, 2:] = mcmtmp[:-2,:-2]
    return res

    
def _mcm00_ducc_tri(spec, lmax,nthreads):
    out= np.empty((spec.shape[0],1,((lmax+1)*(lmax+2))//2),dtype=np.float32)
    ducc0.misc.experimental.coupling_matrix_spin0and2_tri(spec.reshape((spec.shape[0],1,spec.shape[1])), lmax, (0,0,0,0), (0,-1,-1,-1,-1), nthreads=nthreads, res=out)
    return out

def _mcm02_ducc_tri(spec, lmax,nthreads):
    out= np.empty((spec.shape[0],5,((lmax+1)*(lmax+2))//2),dtype=np.float32)
    ducc0.misc.experimental.coupling_matrix_spin0and2_tri(spec[:,:,:], lmax, (0,1,2,3), (0,1,2,3,4), nthreads=nthreads, res=out)
    return out

def _mcmpm_ducc_tri(spec, lmax,nthreads):
    out= np.empty((spec.shape[0],2,((lmax+1)*(lmax+2))//2),dtype=np.float32)
    ducc0.misc.experimental.coupling_matrix_spin0and2_tri(spec[:,3:,:], lmax, (0,0,0,0), (-1,-1,-1,0,1), nthreads=nthreads, res=out)
    return out

def _mcm02_pure_ducc(spec, lmax,nthreads):
    res = np.empty((nspec, 4, lmax+1, lmax+1), dtype=np.float32)
    return ducc0.misc.experimental.coupling_matrix_spin0and2_pure(spec, lmax, nthreads=nthreads, res=res)

# Modified version of ducc0 mcm_bench.py
class Benchmark(object):
    def __init__(self,lmax, bin_edges = None,nthreads=None,verbose=False):
        self.lmax = lmax
        self.nthreads = nthreads
        # number of spectra to process simultaneously
        nspec=1
        if verbose:
            print()
            print("Mode coupling matrix computation comparison")
            print(f"nspec={nspec}, lmax={lmax}, nthreads={nthreads}")
        # we generate the spectra up to 2*lmax+1 to use all Wigner 3j symbols
        # but this could also be lower.
        seed = 1
        np.random.seed(seed)
        cls = np.random.normal(size=(2*lmax+1,))
        self.spec = np.repeat(cls[None, :], repeats=4, axis=0)[None,...]


    def get_mcm(self,code,spin=0,bin_edges=None,bin_weights=None):

        a = time()
        if bin_edges is not None:
            nbins = len(bin_edges)-1
        if code=='ducc' or code=='pspy' or code=='nmt':
            if spin==0:
                if code=='pspy':
                    f = _mcm00_pspy
                elif code=='nmt':
                    f = _mcm00_nmt
                elif code=='ducc':
                    f = lambda x, y: _mcm00_ducc_tri(x,y,nthreads=self.nthreads)
                ducc = f(self.spec[:,0,:], self.lmax)
                if code=='ducc':
                    mcm = _tri2full(ducc, self.lmax)[:,0,:,:][0]
                else:
                    mcm = ducc[:,:,:][0]
            elif spin==2:
                if code!='ducc': raise ValueError
                duccpm = _mcmpm_ducc_tri(self.spec, self.lmax,nthreads=self.nthreads)
                mcmi = _tri2full(duccpm, self.lmax)[0]
            if bin_edges is not None:
                if spin==0:
                    mcm = bin_square_matrix(mcm,bin_edges,self.lmax,bin_weights=bin_weights)
                elif spin==2:
                    mcm = np.zeros((2,nbins,nbins))
                    for i in range(2):
                        mcm[i] = bin_square_matrix(mcmi[i],bin_edges,self.lmax,bin_weights=bin_weights)
            else:
                if spin==0:
                    mcm = mcm[2:,2:]
                elif spin==2:
                    mcm = mcmi[:,2:,2:]

        elif code=='wiggle':
            if spin==0:
                mcm = pywiggle.get_coupling_matrix_from_mask_cls(self.spec[0,0],self.lmax,spintype='TT',
                                                                     bin_edges = bin_edges,bin_weights = bin_weights,verbose=False)
            elif spin==2:
                mcm = pywiggle.get_coupling_matrix_from_mask_cls(self.spec[0,0],self.lmax,spintype=['+','-'],
                                                                 bin_edges = bin_edges,bin_weights = bin_weights,verbose=False)

            if bin_edges is None:
                if spin==0:
                    mcm = mcm[2:,2:]
                elif spin==2:
                    mcm = mcm[:,2:,2:]
                
        else:
            raise ValueError

        etime = time()-a

        return mcm,etime

