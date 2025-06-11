import numpy as np
import warnings
import fastgl
try:
    from pixell.curvedsky import alm2cl
except:
    from healpy import alm2cl
from . import _wiggle
from .utils import multipole_to_bin_indices, bin_array, normalize_weights_per_bin, _parity_flip
import healpy as hp

from functools import cache
import os

_reserved_bin_id = '__unity'


"""
=========
Core classes and functions
=========
"""


class Wiggle(object):
    """
    Gauss-Legendre Power Spectrum Mode-Decoupler
    ============================================

    This class provides the numerical backend for computing mode-decoupled 
    angular power spectra using Gauss-Legendre quadrature methods. It is 
    particularly suited for pseudo-spectrum estimation in the presence of 
    incomplete sky coverage (e.g., masked observations in CMB and LSS analyses).

    Parameters
    ----------
    lmax : int
        Maximum multipole to consider in the analysis.

    bin_edges : array_like, optional
        Array of bin edges in multipole space for binning power spectra.
        If provided, the spectrum will be binned accordingly. Binned
        calculations are significantly faster. Note, these are of the form
        low_edge <= ℓ < upper_edge.


    verbose : bool, default=True
        Whether to print verbose information during computation.

    xlmax : int, default=2
        Controls how far in multipole space to use mask pseudo-spectra.
        Must be ≥2 for accurate decoupling, i.e. mask spectra need to
        provided for at least 2 x lmax.

    xgllmax : int, default=2
        Multiplier controlling the number of Gauss-Legendre quadrature points 
        used, relative to lmax. Should typically be ≥2 for accurate integrals.


    Notes
    -----
    - The object precomputes the Legendre matrix used for spin-0 correlations 
      (`cd00`), as well as a view truncated to `lmax` (`ud00`).
    - The internal memory use increases with the number of masks or beams added.
    - All internal 2D matrices (e.g., cd00) are stored in `(theta, ell)` order.
    """
    def __init__(self,lmax,
                 bin_edges = None,
                 verbose = True, xlmax = 2, xgllmax = 2):
        
        # TODO: cache_file implementation
        self._spintype = {'TT':'00','TE':'20','EE':'++','BB':'--'}
            
        if xlmax<2:
            warnings.warn("Not using mask Cls out to 2*lmax. Mode decoupling may not be accurate!")
        if xgllmax<2:
            warnings.warn("Not using at least 2*lmax+1 GL quadrature points. Mode decoupling may not be accurate!")
        self.xlmax = xlmax
        self.xgllmax = xgllmax
        self.verbose = verbose
        self.lmax = lmax
        self.Nlmax = (xgllmax*lmax+1) # GL number of weights
        self.ells = np.arange(xlmax*lmax+1)
        self.mu, self.w_mu = fastgl.roots_legendre(self.Nlmax) 
        # This is d00 = P_ell evaluated on 2lmax x 2lmax
        # It always needs to be computed since it is needed for correlation
        # functions, so might as well do it now. It has to be unbinned.
        self.cd00 = _wiggle._compute_legendre_matrix(xlmax*lmax,self.mu)
        

        # Binning prep if needed
        self._binned = False
        if bin_edges is not None:
            self.nbins = len(bin_edges)-1
            self._bin_indices = multipole_to_bin_indices(lmax, bin_edges)
            self._binned = True
            

            
        
        # This is needed for spin-0; no extra cost in getting a view of it,
        self.ud00 = self.cd00[:,:lmax+1]

        self._mask_cl_cache = {}
        self._mask_alm_cache = {}
        self._nweights = {}
        self._beam_fl = {}


    @cache
    def _get_corr(self,mask_id1,mask_id2,parity):
        if mask_id2 is None: mask_id2 = mask_id1
        # this gives me a parity weighted correlation function from mask Cls
        coeff = (2*self.ells+1)/(4*np.pi) * self._get_mask_cls(mask_id1,mask_id2=mask_id2)[:(self.xlmax*self.lmax+1)] # (2*lmax+1,)
        coeff = _parity_flip(coeff,parity) # this applies (-1)^ell if parity is '-'
        # cd00 are just Legendre polynomials (or Wigner d_00)
        xi = self.cd00 @ coeff # (N,)
        return xi

    @cache
    def _get_G_term_weights(self,mode_count_weight,parity,apply_bin_weights,bin_weight_id):
        # Weights needed for the the things that multiply Wigner-ds in the G-matrices
        # Start building weights
        ws = np.ones(self.lmax+1)
        ws = _parity_flip(ws,parity)  # this applies (-1)^ell if parity is '-'
        if mode_count_weight:
            ws = ws * ((2*self.ells[:self.lmax+1]+1)/2.)
        if self._binned and apply_bin_weights:
            ws = ws * self._nweights[bin_weight_id]
        return ws

    @cache
    def _get_b1_b2(self,spin1,spin2,parity,bin_weight_id,
                   beam_id1,beam_id2):
        # Get the left (ell) and right (ell') sides of the G-matrices
        
        # these are weights that multiply the ell side of the G matrix
        # it includes a parity term as well bandpower bin weights
        nweights = self._get_G_term_weights(mode_count_weight=False,parity=parity,apply_bin_weights=True,bin_weight_id=bin_weight_id)

        
        # these are weights that multiply the ell' side of the G matrix
        # it includes a parity term and a mode counting term but no bin weights
        nweights2 = self._get_G_term_weights(mode_count_weight=True,parity=parity,apply_bin_weights=False,bin_weight_id=None)
        # the ell' side also includes beam transfers if you want to deconvolve those
        if beam_id1 is not None:
            nweights2 = nweights2 * self._beam_fl[beam_id1]
        if beam_id2 is not None:
            nweights2 = nweights2 * self._beam_fl[beam_id2]

        if self._binned:
            # we efficiently calculate the binned ell and ell' sides by binning the weights times Wigner-ds
            b1,b2 = _wiggle._compute_double_binned_wigner_d(self.lmax,spin1,spin2,self.mu,
                                                            self.nbins,self._bin_indices,nweights,nweights2)
        else:
            # in the unbinned case we simply multiply the weights and wigner-ds
            if (spin1==0) and (spin2==0): ud = self.ud00
            if (spin1==2) and (spin2==2):
                ud = self._get_wigner(2,2)
            if (spin1==2) and (spin2==0):
                ud = self._get_wigner(2,0)
            b1 = nweights[None,:] * ud
            b2 = nweights2[None,:] * ud

        return b1,b2

    @cache
    def _get_wigner(self,spin1,spin2):
        return _wiggle._compute_wigner_d_matrix(self.lmax,spin1,spin2,self.mu)
        
    @cache
    def _get_m(self,mask_id1,mask_id2,spin1,spin2,parity,bin_weight_id,
               beam_id1,beam_id2):
        # Get the core of the mode-coupling G matrices. This is where the expensive
        # dot product happens.

        # Know of situations where other spins would be useful? Write to us!
        # TODO: Implement pure E/B terms which require spin-1
        if [spin1,spin2] not in [[0,0],[2,2],[2,0]]: raise NotImplementedError

        # Get the left (ell) and right (ell') sides of the G-matrices
        b1,b2 = self._get_b1_b2(spin1,spin2,parity,bin_weight_id=bin_weight_id,beam_id1=beam_id1,beam_id2=beam_id2)
            
        # Get the Gauss-Legendre quadrature weighted correlation functions of the mask
        xi = self._get_corr(mask_id1,mask_id2,parity=parity)
        W = self.w_mu * xi
        
        # Know how to speed up matrix products? Write to us! (This one does use OpenMP threads)
        M = np.einsum('i,ij,ik->jk', W, b1, b2, optimize='greedy')
        return M
    
    def _get_coupling_matrix_from_ids(self,mask_id1,mask_id2,spintype,bin_weight_id,
                                          beam_id1,beam_id2):
        # Get mode-coupling G matrices
        
        if spintype not in ['00','++','--','20']: raise ValueError
        f = lambda spin1,spin2,parity: self._get_m(mask_id1,mask_id2,spin1=spin1,spin2=spin2,parity=parity,
                                              bin_weight_id=bin_weight_id,
                                              beam_id1=beam_id1,beam_id2=beam_id2)
        if spintype=='00':
            return f(0,0,'+')
        elif spintype in ['++','--']:
            g1 = f(2,2,'+')
            g2 = f(2,2,'-')
            if spintype=='++':
                return (g1+g2)/2.
            elif spintype=='--':
                return (g1-g2)/2.
        elif spintype=='20':
            return f(2,0,'+')

    def _get_mask_cls(self,mask_id1,mask_id2):
        if mask_id2 is None: mask_id2 = mask_id1
        try:
            mcls = self._mask_cl_cache[f'{mask_id1}_{mask_id2}']
            if self.verbose: print(f"Reusing mask cls {mask_id1}_{mask_id2}...")
            return mcls
        except KeyError:
            self._mask_cl_cache[f'{mask_id1}_{mask_id2}'] = alm2cl(self._mask_alm_cache[mask_id1],self._mask_alm_cache[mask_id2])
            return self._mask_cl_cache[f'{mask_id1}_{mask_id2}']

    def _thfilt_core(self,mask_id1,mask_id2,spin1,spin2,parity,bin_weight_id,beam_id1,beam_id2):
        if spin1==0 and spin2==0:
            ud = self.ud00
        elif spin1==2 and spin2==0:
            ud = self._get_wigner(2,0)
        elif spin1==2 and spin2==2:
            ud = self._get_wigner(2,2)
        b1,b2 = self._get_b1_b2(spin1,spin2,parity,bin_weight_id=bin_weight_id,beam_id1=beam_id1,beam_id2=beam_id2) # (N x nbins)
        xi = self._get_corr(mask_id1,mask_id2,parity=parity)
        W = self.w_mu * xi # (N,)
        b2w = self._get_G_term_weights(mode_count_weight=True,parity=parity,apply_bin_weights=False,bin_weight_id=None) # (nells,)
        R2 = ud * b2w[None,:] # (N, nells)
        M = np.einsum('i,ij,ik->jk', W, b1, R2, optimize='greedy') # (nbins,nells)
        return M
        
        
    @cache
    def get_theory_filter(self,mask_id1,mask_id2=None,spintype='00',bin_weight_id=None,beam_id1=None,beam_id2=None):
        r"""
        Construct the theoretical bandpower filter :math:`\mathcal{F}^{s_as_b}_{q\ell}`

        This method computes the filter matrix that transforms the theoretical full-sky power 
        spectrum :math:`\mathsf{C}^{ab,\mathrm{th}}_\ell` into the corresponding prediction 
        for the *binned, decoupled* bandpowers :math:`\mathsf{B}^{ab,\mathrm{th}}_q` in the 
        presence of mode coupling and nontrivial binning. The filter is given by:

        .. math::
            \mathrm{vec}\left[\mathsf{B}^{ab,\mathrm{th}}_q\right] =
            \sum_{\ell} \mathcal{F}^{s_as_b}_{q\ell} \cdot 
            \mathrm{vec}\left[\mathsf{C}^{ab,\mathrm{th}}_\ell\right],

        where:

        .. math::
            \mathcal{F}^{s_as_b}_{q\ell} =
            \sum_{q'} \left(\mathcal{M}^{s_as_b}\right)^{-1}_{qq'} 
            \sum_{\ell' \in \vec{\ell}_{q'}} w_{q'}^{\ell'} 
            \mathsf{M}^{s_as_b}_{\ell'\ell}.


        Parameters
        ----------
        mask_id1 : str
            Identifier for the first mask used in computing the coupling matrices, previously provided through `self.add_mask`.

        mask_id2 : str or None, optional
            Identifier for the second mask (e.g., in cross-spectra), previously provided through `self.add_mask`. If `None`, 
            defaults to `mask_id1`.

        spintype : str, default='00'
            Specifies the spin combination of the fields:
            - `'00'` for scalar × scalar (e.g., temperature or κ)
            - `'++'`, `'--'` for E/B-mode combinations (spin-2 × spin-2)
            - `'20'` for spin-2 × spin-0 cross spectra (e.g., shear × κ)

        bin_weight_id : str or None, optional
            ID of the binning weights to use. If not provided, defaults to uniform binning.

        Returns
        -------
        thfilt : ndarray
            A matrix of shape `(n_bins, lmax + 1)` representing the filter 
            :math:`\mathcal{F}^{s_as_b}_{q\ell}` to apply to theory spectra for direct 
            comparison with bandpower estimates.

        """
        
        if mask_id2 is None: mask_id2 = mask_id1
        f = lambda spin1, spin2, parity: self._thfilt_core(mask_id1,mask_id2,spin1,spin2,parity,bin_weight_id=bin_weight_id,
                                                      beam_id1=beam_id1,beam_id2=beam_id2)
        if spintype=='00':
            Mc = f(0,0,'+')
        elif spintype in ['++','--']:
            Mc1 = f(2,2,'+')
            Mc2 = f(2,2,'-')
            if spintype=='++':
                Mc = (Mc1+Mc2)/2.
            elif spintype=='--':
                Mc = (Mc1-Mc2)/2.
        elif spintype=='20':
            Mc = f(2,0,'+')
        cinv = self._get_cinv(mask_id1,mask_id2=mask_id2,spintype=spintype,bin_weight_id=bin_weight_id,
                              beam_id1=beam_id1,beam_id2=beam_id2)
        thfilt = np.einsum('ij,jk->ik', cinv, Mc, optimize='greedy')
        return thfilt

    
    def add_mask(self, mask_id, mask_alms=None, mask_cls=None):
        r"""
        Register a mask for use in mode-coupling and decoupling calculations.

        This method adds a sky mask to the internal cache, either in spherical harmonic 
        (`alm`) form or as a precomputed pseudo-power spectrum (`Cl`). The mask is used 
        to compute mode-coupling matrices that correct for incomplete sky coverage.

        Parameters
        ----------
        mask_id : str
            A unique string identifier for the mask. This ID will be used in subsequent 
            calls that reference masks for pseudo-Cl computation or coupling matrix generation.

        mask_alms : array_like, optional
            Spherical harmonic coefficients (1D array) of the mask map. Must have sufficient 
            resolution, i.e., support at least `xlmax * lmax` in multipole space. Required 
            if `mask_cls` is not provided.

        mask_cls : array_like, optional
            Pseudo-`Cl` spectrum of the mask, used as a shortcut to avoid computing 
            `mask_alms`. Must cover multipoles up to `xlmax * lmax`. Cannot be provided 
            at the same time as `mask_alms`.

        Raises
        ------
        ValueError
            - If both `mask_alms` and `mask_cls` are provided.
            - If `mask_alms` is multidimensional (should be a 1D array).
            - If the resolution of `mask_alms` or `mask_cls` is insufficient for the 
              configured `xlmax * lmax`.

        Notes
        -----
        - If `mask_cls` is provided, it is stored directly and the harmonic coefficients 
          are not needed.
        - If `mask_alms` is provided, its `lmax` is checked against the required cutoff 
          for accurate mode-coupling computation.
        - The mask is cached internally using the specified `mask_id` and can be reused 
          in auto- and cross-spectrum computations.
        """
        
        if mask_cls is not None:
            if mask_alms is not None: raise ValueError
            self._mask_cl_cache[f'{mask_id}_{mask_id}'] = mask_cls[:self.xlmax*self.lmax+1]
            return
        if mask_alms.ndim>1: raise ValueError
        lmax = hp.Alm.getlmax(mask_alms.size)
        if lmax<(self.xlmax*self.lmax): raise ValueError(f"Mask Cls need to be at least {self.xlmax} x lmax. Calculate mask SHTs out to higher ell or consider lowering xlmax in the initialization (not recommended!).")
        self._mask_alm_cache[mask_id] = mask_alms


    @cache
    def _get_cinv(self,mask_id1,mask_id2,spintype,bin_weight_id,
                  beam_id1,beam_id2):
        mcm = self._get_coupling_matrix_from_ids(mask_id1,mask_id2,spintype=spintype,
                                                 beam_id1=beam_id1,beam_id2=beam_id2,bin_weight_id=bin_weight_id)
        cinv = np.linalg.inv(mcm)
        return cinv
    
    def _populate_unity_bins(self,):
        bin_weight_id = _reserved_bin_id
        if _reserved_bin_id not in self._nweights.keys():
            self._nweights[_reserved_bin_id] = normalize_weights_per_bin(self.nbins, self._bin_indices, np.ones(self.lmax+1))

    def add_bin_weights(self,weight_id,bin_weights):
        r"""
        Register custom binning weights for multipole binning.

        This method allows the user to provide a weighting scheme when projecting 
        multipole spectra into bandpowers. This is commonly used to apply inverse-variance 
        weighting or to mimic a specific theoretical spectrum shape during the binning 
        operation. The weights are normalized within each bin and stored under a user-defined ID.

        Parameters
        ----------
        weight_id : str
            A unique string identifier for the binning weights. Must not use reserved internal IDs.

        bin_weights : array_like or None
            An array of weights of shape `(lmax + 1,)`. Each element corresponds to a weight 
            for the multipole :math:`\ell`, starting at 0. If `None` is passed, the method falls back to using 
            uniform weights within each bin.

        Raises
        ------
        ValueError
            If `weight_id` is reserved for internal use.
            If `bin_weights` is provided but does not cover at least up to `lmax`.

        Notes
        -----
        - The weights are automatically normalized within each bin.
        - Only the first `lmax + 1` values are used.
        - Once registered, the weights can be referred to by their `weight_id` in calls to 
          power spectrum estimation methods.
        """
        if not(self._binned): return
        if weight_id==_reserved_bin_id: raise ValueError("That ID is reserved for internal use.")
        if bin_weights is None:
            bin_weights = np.ones(self.lmax+1)
            
        if bin_weights.size<(self.lmax+1): raise ValueError
        bin_weights = bin_weights[:self.lmax+1]
        nweights = normalize_weights_per_bin(self.nbins, self._bin_indices, bin_weights)
        self._nweights[weight_id] = nweights

    def add_beam(self, beam_id, beam_fl):
        r"""
        Register a beam transfer function for use in power spectrum beam deconvolution.

        This method adds a 1D multiplicative beam transfer function :math:`B_\ell` 
        that can be deconvolved from power spectra. 

        Parameters
        ----------
        beam_id : str
            A unique string identifier for the beam. This ID will be referenced in calls 
            to decoupling or filtering methods that support beam correction.

        beam_fl : array_like
            A 1D array of shape `(lmax + 1,)` or larger, specifying the multiplicative 
            filter to apply to each multipole :math:`\ell`, starting at zero. Only the first `lmax + 1` values 
            will be retained.

        Raises
        ------
        ValueError
            If the length of `beam_fl` is less than `lmax + 1`, indicating insufficient 
            multipole support for the configured maximum multipole.

        Notes
        -----
        - Multiple beams can be registered simultaneously under different IDs and used 
          in cross-spectrum configurations.
        """
        
        if beam_fl.size < (self.lmax+1): raise ValueError(f"Beam filter need to be at least lmax.")
        self._beam_fl[beam_id] = beam_fl[:self.lmax+1]
        
    def decoupled_cl(self,alms1,alms2, mask_id1, mask_id2=None, spectype='TT',
                     bin_weight_id = None,
                     beam_id1 = None, beam_id2 = None,
                     return_theory_filter=False):

        r"""
        Compute decoupled angular power spectra (:math:`C_{\ell}`) from the spherical harmonics
        of maps that have already been masked.

        This method estimates the angular power spectrum between two input fields 
        in harmonic space (`alm`s), accounting for mode coupling due to partial sky coverage,
        beam smoothing, and bandpower binning. The result is a debiased, decoupled bandpower 
        estimate suitable for direct comparison with theoretical predictions. Note that
        a theory filter of shape (nbins,nells) needs to be applied to (nells,) shaped
        theory spectra if using bandpower binning. This filter can be obtained from this
        function call, but is the most expensive part of the calculation, so consider
        obtaining it from `self.get_theory_filter` once.

        Parameters
        ----------
        alms1 : array_like
            Spherical harmonic coefficients of the first map (e.g., T, E, or B, galaxy overdensity, convergence).

        alms2 : array_like
            Spherical harmonic coefficients of the second map. Can be the same as `alms1`
            for auto-spectra.

        mask_id1 : str
            Identifier for the mask applied to the first field, previously provided through `self.add_mask`.

        mask_id2 : str or None, optional
            Identifier for the mask on the second field, previously provided through `self.add_mask`. If `None`, uses `mask_id1`.

        spectype : str, default='TT'
            Type of power spectrum to compute. Supported values are:
            - `'TT'` (both alms are spin-0 scalars, e.g. for CMB temperature, galaxy overdensity, convergence)
            - `'EE'` (both alms are E-mode decompositions of a spin-2 field)
            - `'BB'` (both alms are B-mode decompositions of a spin-2 field)
            - `'TE'` (alms are a spin-0 scalar and an E-mode decomposition of a spin-2 field)

        bin_weight_id : str or None, optional
            Identifier for custom multipole binning weights. If not specified,
            unity weights will be used by default.

        beam_id1 : str or None, optional
            Beam ID for the first map to deconvolve a beam previously provided through `self.add_beam`.
            If None, no beam is deconvolved for the first map.

        beam_id2 : str or None, optional
            Beam ID for the second map to deconvolve a beam previously provided through `self.add_beam`.
            If None, no beam is deconvolved for the second map.

        return_theory_filter : bool, default=False
            If True, also return the theory bandpower filter 
            :math:`\mathcal{F}^{s_as_b}_{q\ell}` for use in model comparison.

        Returns
        -------
        dcls : ndarray
            The decoupled, binned power spectrum as a 1D array over bandpowers.

        tdecmat : ndarray, optional
            The theory filter matrix :math:`\mathcal{F}^{s_as_b}_{q\ell}` of shape 
            `(n_bins, lmax + 1)` if `return_theory_filter` is True.

        Notes
        -----
        - This method is the recommended interface for evaluating both auto- and 
          cross-spectra in masked sky analyses, but other convenience wrappers are provided elsewhere.
        """
        if spectype not in ['TT','EE','BB','TE']: raise NotImplementedError

        # Unity bin weights if none specified
        if self._binned and (bin_weight_id is None):
            self._populate_unity_bins()
                
        # Get MCM
        cinv = self._get_cinv(mask_id1,mask_id2=mask_id2,spintype=self._spintype[spectype],
                              bin_weight_id=bin_weight_id,
                              beam_id1=beam_id1,beam_id2=beam_id2)
        field_cls = alm2cl(alms1,alms2)[:self.lmax+1]
        # Bin Cls
        if self._binned:
            bcls = bin_array(field_cls, self._bin_indices, self.nbins,weights=self._nweights[bin_weight_id])
        else:
            bcls = field_cls
        # Decouple
        dcls = np.dot(cinv,bcls)
        
        if return_theory_filter:
            tdecmat = self.get_theory_filter(mask_id1,mask_id2,spintype=self._spintype[spectype],
                                             bin_weight_id=bin_weight_id,
                                             beam_id1=beam_id1,beam_id2=beam_id2)
            return dcls, tdecmat
        else:
            return dcls


def get_coupling_matrix_from_mask_cls(mask_cls,lmax,spintype='00',bin_edges = None,bin_weights = None,
                                      beam_fl1 = None,beam_fl2 = None,
                                      return_obj=False):
    r"""
    Compute the binned mode-coupling matrix from the pseudo-Cl of a sky mask.

    This function is a high-level wrapper to generate the binned mode-coupling matrix 
    :math:`\mathcal{M}^{s_as_b}_{q q'}` using the pseudo-power spectrum of a mask. 
    This matrix describes how true sky power at one multipole leaks into others due to 
    incomplete sky coverage, beam smoothing, and binning.

    Parameters
    ----------
    mask_cls : array_like
        Pseudo-Cl power spectrum of the mask, covering multipoles up to at least 
        `2 * lmax`, starting at 0. This should be precomputed externally.

    lmax : int
        Maximum multipole for the output coupling matrix.

    spintype : str, default='00'
        Spin combination of the fields:
        - `'00'`: scalar × scalar (e.g., T × T or κ × κ)
        - `'++'`, `'--'`: spin-2 × spin-2 (e.g., E × E or B × B)
        - `'20'`: spin-2 × spin-0 (e.g., E × κ or γ × κ)

    bin_edges : array_like, optional
        Array of bin edges defining bandpowers. If not provided, no binning is applied.
        Note, these are of the form low_edge <= ℓ < upper_edge.

    bin_weights : array_like, optional
        Weights for each multipole used in binning. Must have at least `lmax + 1` entries, starting at 0.
        If not provided, uniform weights will be assumed.

    beam_fl1 : array_like, optional
        Beam transfer function for the first field (length ≥ `lmax + 1`). Optional.

    beam_fl2 : array_like, optional
        Beam transfer function for the second field. Optional.

    return_obj : bool, default=False
        If True, also return the internal `Wiggle` object for further manipulation.

    Returns
    -------
    m : ndarray
        The binned mode-coupling matrix of shape `(n_bins, n_bins)`.

    g : Wiggle, optional
        The `Wiggle` object used to generate the matrix, returned only if `return_obj=True`.

    """
    g = Wiggle(lmax,bin_edges = bin_edges)
    g.add_mask('m1',mask_cls=mask_cls)
    g.add_bin_weights('b1',bin_weights = bin_weights)
    if beam_fl1 is not None:
        g.add_beam('p1',beam_fl1)
        beam_id1 = 'p1'
    else:
        beam_id1 = None
    if beam_fl2 is not None:
        g.add_beam('p2',beam_fl2)
        beam_id2 = 'p2'
    else:
        beam_id2 = None
    m = g._get_coupling_matrix_from_ids('m1','m1',spintype=spintype,bin_weight_id='b1',
                                       beam_id1=beam_id1,beam_id2=beam_id2)
    if return_obj:
        return m,g
    return m
    
def alm2auto_power_spin0(lmax,alm,mask_alm,bin_edges = None,bin_weights = None,beam_fl=None,
                    return_theory_filter=True):
    r"""
    Compute the decoupled auto power spectrum for a spin-0 field (e.g., CMB temperature or lensing convergence).

    This function estimates the angular power spectrum :math:`C_\ell` from a masked 
    spherical harmonic map (`alm`) of a scalar (spin-0) field, such as CMB temperature (T) 
    or lensing convergence (κ). It accounts for the effects of partial sky coverage, 
    beam smoothing, and optional binning.

    Parameters
    ----------
    lmax : int
        Maximum multipole for the analysis.

    alm : array_like
        Spherical harmonic coefficients of the input scalar map.

    mask_alm : array_like
        Spherical harmonic coefficients of the mask applied to the map. Must be defined 
        up to 2 x lmax to support accurate mode coupling corrections.

    bin_edges : array_like, optional
        Array defining the edges of multipole bins (e.g., for bandpowers). If omitted,
        returns unbinned Cl estimates. Note, these are of the form low_edge <= ℓ < upper_edge.

    bin_weights : array_like, optional
        Weights applied to multipoles within each bin. Must be at least length `lmax + 1`, starting at zero.

    beam_fl : array_like, optional
        Beam transfer function :math:`B_\ell` to apply to the theory and/or deconvolve 
        from the data. Must be at least length `lmax + 1`, starting at and normalized to `\ell=0`.

    return_theory_filter : bool, default=True
        If True, also return the theory filter matrix 
        :math:`\mathcal{F}^{00}_{q\ell}` to apply to full-sky theory spectra 
        for bandpower comparison.

    Returns
    -------
    dcls : ndarray
        The decoupled (and optionally binned) auto power spectrum :math:`\hat{C}_\ell`.

    tdecmat : ndarray, optional
        The theory filter matrix of shape `(n_bins, lmax + 1)` that maps full-sky theory 
        Cls to binned, decoupled bandpowers. Returned only if `return_theory_filter=True`.

    Notes
    -----
    - This method uses the fast Gauss-Legendre pseudo-`Cl` formalism with accurate 
      mode-decoupling and beam correction.
    - It is appropriate for scalar fields with no spin (e.g., CMB temperature, galaxy overdensity, κ).
    - Use this as the primary interface for auto-power spectrum estimation of scalar quantities.
    """
    
    g = Wiggle(lmax,bin_edges = bin_edges)
    g.add_mask('m1',mask_alm)
    g.add_bin_weights('b1',bin_weights = bin_weights)
    if beam_fl is not None:
        g.add_beam('p1',beam_fl)
        beam_id = 'p1'
    else:
        beam_id = None

    return g.decoupled_cl(alm,alm,'m1','m1',spectype='TT',
                          bin_weight_id = 'b1',
                          beam_id1 = beam_id, beam_id2 = beam_id,
                          return_theory_filter=return_theory_filter)

def alm2cross_power_spin0(lmax,alm1,alm2,mask_alm,mask_alm2=None,bin_edges = None,bin_weights = None,beam_fl=None,
                          beam_fl2=None,
                    return_theory_filter=True):
    r"""
    Compute the decoupled cross power spectrum for two spin-0 fields (e.g., T_1 × T_2, κ × δ_g).

    This function estimates the angular cross power spectrum :math:`C_\ell^{12}` between 
    two masked spin-0 spherical harmonic maps (`alm1` and `alm2`), such as CMB temperature maps at different frequencies
    or lensing convergence and galaxy overdensity. It corrects for sky masking, beam effects, and optional binning, 
    returning a debiased, decoupled bandpower estimate.

    Parameters
    ----------
    lmax : int
        Maximum multipole for the analysis.

    alm1 : array_like
        Spherical harmonic coefficients of the first scalar field.

    alm2 : array_like
        Spherical harmonic coefficients of the second scalar field.

    mask_alm : array_like
        Spherical harmonic coefficients of the mask applied to `alm1`. Should be defined
        up to `2 * lmax`.

    mask_alm2 : array_like or None, optional
        Mask for `alm2`. If None, defaults to `mask_alm`.

    bin_edges : array_like, optional
        Array defining the bin edges in multipole space for bandpower estimation.
        Note, these are of the form low_edge <= ℓ < upper_edge.

    bin_weights : array_like, optional
        Weights applied to multipoles within each bin. Must be at least `lmax + 1` in length,
        starting from :math:`\ell=0`.

    beam_fl : array_like, optional
        Beam transfer function :math:`B_\ell^{(1)}` for the first field. Must be at least 
        `lmax + 1` in length and normalized at :math:`\ell=0`.

    beam_fl2 : array_like, optional
        Beam transfer function :math:`B_\ell^{(2)}` for the second field. Same requirements as `beam_fl`.

    return_theory_filter : bool, default=True
        If True, also return the theory filter matrix 
        :math:`\mathcal{F}^{00}_{q\ell}` that projects full-sky theory spectra into 
        bandpower space for masked-sky comparison.

    Returns
    -------
    dcls : ndarray
        The decoupled (and optionally binned) cross power spectrum :math:`\hat{C}_\ell^{12}`.

    tdecmat : ndarray, optional
        The theory filter matrix of shape `(n_bins, lmax + 1)` for comparing 
        full-sky theory to decoupled bandpowers. Returned only if `return_theory_filter=True`.

    Notes
    -----
    - This function uses the Gauss-Legendre pseudo-`Cl` formalism with full treatment 
      of beam convolution, mode-coupling, and binning.
    """
    g = Wiggle(lmax,bin_edges = bin_edges)
    g.add_mask('m1',mask_alm)
    if mask_alm2 is not None:
        g.add_mask('m2',mask_alm2)
    g.add_bin_weights('b1',bin_weights = bin_weights)
    if beam_fl is not None:
        g.add_beam('p1',beam_fl)
        beam_id1 = 'p1'
    else:
        beam_id1 = None
    if beam_fl2 is not None:
        g.add_beam('p2',beam_fl2)
        beam_id2 = 'p2'
    else:
        beam_id2 = None

    return g.decoupled_cl(alm1,alm2,'m1','m2' if mask_alm2 is not None else 'm1',spectype='TT',
                          bin_weight_id = 'b1',
                          beam_id1 = beam_id1, beam_id2 = beam_id2,
                          return_theory_filter=return_theory_filter)

def alm2auto_power_spin2(lmax, alm_e, alm_b, mask_alm, bin_edges=None, bin_weights=None,
                         beam_fl_e=None, beam_fl_b=None,
                         return_theory_filter=True):
    r"""
    Compute decoupled auto-power spectra (EE and/or BB) for spin-2 spherical harmonic maps 
    using masked sky observations.

    This function estimates the E-mode and B-mode angular power spectra from 
    input spherical harmonic coefficients (`alm_e`, `alm_b`), accounting for sky masking,
    beam smoothing, and optional binning. The result is a debiased, decoupled
    bandpower estimate suitable for comparison with theory predictions in 
    cosmological analyses (e.g., CMB or weak lensing). Either alm_e or alm_b can be None
    if either EE or BB are not needed.

    Parameters
    ----------
    lmax : int
        Maximum multipole (`ell`) up to which the power spectrum is computed.

    alm_e : array_like or None
        Spherical harmonic coefficients of the E-mode polarization map.
        If `None`, EE spectrum will not be computed.

    alm_b : array_like or None
        Spherical harmonic coefficients of the B-mode polarization map.
        If `None`, BB spectrum will not be computed.

    mask_alm : array_like
        Spherical harmonic coefficients of the mask that was applied to the spin-2 fields to obtain the E and B fields. Should be defined
        up to `2 * lmax`.

    bin_edges : array_like, optional
        Multipole bin edges for bandpower binning. If not provided, the output spectra 
        will not be binned. Note, these are of the form low_edge <= ℓ < upper_edge.

    bin_weights : array_like, optional
        Weights applied to each multipole within a bin. Useful for inverse-variance or 
        custom binning schemes.

    beam_fl_e : array_like or None, optional
        Beam transfer function for the E-mode map. If provided, the beam will be deconvolved.

    beam_fl_b : array_like or None, optional
        Beam transfer function for the B-mode map. If provided, the beam will be deconvolved.

    return_theory_filter : bool, default=True
        If `True`, also returns the theory bandpower filter matrix 
        :math:`\mathcal{F}^{++}_{q\ell}` and :math:`\mathcal{F}^{--}_{q\ell}` 
        for EE and BB respectively, which can be used to convolve theoretical spectra
        for comparison.

    Returns
    -------
    cl_EE : ndarray or None
        Decoupled EE auto-power spectrum as a 1D array over bins. Returned only if `alm_e` is not None, otherwise None is returned.

    tf_EE : ndarray or None
        Theory bandpower filter matrix for EE. Returned only if `alm_e` is not None and 
        `return_theory_filter` is True.

    cl_BB : ndarray or None
        Decoupled BB auto-power spectrum as a 1D array over bins. Returned only if `alm_b` is not None, otherwise None is returned.

    tf_BB : ndarray or None
        Theory bandpower filter matrix for BB. Returned only if `alm_b` is not None and 
        `return_theory_filter` is True.

    """
    g = Wiggle(lmax, bin_edges=bin_edges)
    g.add_mask('m1', mask_alm)
    g.add_bin_weights('b1', bin_weights=bin_weights)

    beam_ide = 'pe' if beam_fl_e is not None else None
    beam_idb = 'pb' if beam_fl_b is not None else None

    if beam_fl_e is not None:
        g.add_beam(beam_ide, beam_fl_e)
    if beam_fl_b is not None:
        g.add_beam(beam_idb, beam_fl_b)

    if alm_e is not None:
        ret_EE = g.decoupled_cl(alm_e, alm_e, 'm1', spectype='EE',
                                bin_weight_id='b1',
                                return_theory_filter=return_theory_filter,
                                beam_id1=beam_ide, beam_id2=beam_ide)
    else:
        ret_EE = [None,None]

    if alm_b is not None:
        ret_BB = g.decoupled_cl(alm_b, alm_b, 'm1', spectype='BB',
                                bin_weight_id='b1',
                                return_theory_filter=return_theory_filter,
                                beam_id1=beam_idb, beam_id2=beam_idb)
    else:
        ret_BB = [None,None]
    return (*ret_EE, *ret_BB)


def alm2auto_power_spin02(lmax, alm_t, alm_e, alm_b, mask_alm_t, mask_alm_p=None,
                          bin_edges=None, bin_weights=None,
                          beam_fl_t=None, beam_fl_e=None, beam_fl_b=None,
                          return_theory_filter=True):
    r"""
    Compute decoupled auto- and cross-power spectra (TT, EE, BB, TE) for scalar (spin-0) 
    and polarization (spin-2) spherical harmonic maps using masked sky observations, e.g.
    for CMB temperature and polarization, or for LSS scalar fields like convergence and
    galaxy overdensity together with galaxy shear.

    This function estimates the angular power spectra from input spherical harmonic 
    coefficients for a scalar (`alm_t`) and E/B decomposition of a spin-2 (`alm_e`, `alm_b`), correcting 
    for mode coupling induced by sky masking, beam smoothing, and optional multipole binning.
    The output includes debiased, decoupled bandpower spectra suitable for direct 
    comparison with theoretical models.

    Parameters
    ----------
    lmax : int
        Maximum multipole (`ell`) up to which the power spectra are computed.

    alm_t : array_like
        Spherical harmonic coefficients of the scalar (T) map.

    alm_e : array_like
        Spherical harmonic coefficients of the E-mode map.

    alm_b : array_like
        Spherical harmonic coefficients of the B-mode map.

    mask_alm_t : array_like
        Spherical harmonic coefficients of the mask applied to the scalar field. 
        Should be defined up to `2 * lmax`.

    mask_alm_p : array_like or None, optional
        Spherical harmonic coefficients of the mask applied to the spin-2 fields used in the E/B decomposition. If `None`, the scalar mask (`mask_alm_t`) is reused. Should be 
        defined up to `2 * lmax`.

    bin_edges : array_like, optional
        Multipole bin edges for bandpower binning. If not provided, the output spectra 
        will not be binned. Note, these are of the form low_edge <= ℓ < upper_edge.

    bin_weights : array_like, optional
        Weights applied to each multipole within a bin. Useful for inverse-variance or 
        custom binning schemes.

    beam_fl_t : array_like or None, optional
        Beam transfer function for the scalar map. If provided, the beam will be deconvolved.

    beam_fl_e : array_like or None, optional
        Beam transfer function for the E-mode map. If provided, the beam will be deconvolved.

    beam_fl_b : array_like or None, optional
        Beam transfer function for the B-mode map. If provided, the beam will be deconvolved.

    return_theory_filter : bool, default=True
        If `True`, also returns the theory bandpower filter matrices 
        :math:`\mathcal{F}^{s_as_b}_{q\ell}` for TT, EE, BB, and TE, which can be used 
        to convolve theoretical spectra for model comparison.

    Returns
    -------
    cl_TT : ndarray
        Decoupled TT auto-power spectrum as a 1D array over bins.

    tf_TT : ndarray or None
        Theory bandpower filter matrix for TT. Returned only if `return_theory_filter` is True.

    cl_EE : ndarray
        Decoupled EE auto-power spectrum as a 1D array over bins.

    tf_EE : ndarray or None
        Theory bandpower filter matrix for EE. Returned only if `return_theory_filter` is True.

    cl_BB : ndarray
        Decoupled BB auto-power spectrum as a 1D array over bins.

    tf_BB : ndarray or None
        Theory bandpower filter matrix for BB. Returned only if `return_theory_filter` is True.

    cl_TE : ndarray
        Decoupled TE cross-power spectrum as a 1D array over bins.

    tf_TE : ndarray or None
        Theory bandpower filter matrix for TE. Returned only if `return_theory_filter` is True.

    """
    g = Wiggle(lmax, bin_edges=bin_edges)
    g.add_mask('m1', mask_alm_t)
    if mask_alm_p is not None:
        g.add_mask('m2', mask_alm_p)
    g.add_bin_weights('b1', bin_weights=bin_weights)

    beam_idt = 'pt' if beam_fl_t is not None else None
    beam_ide = 'pe' if beam_fl_e is not None else None
    beam_idb = 'pb' if beam_fl_b is not None else None

    if beam_fl_t is not None:
        g.add_beam(beam_idt, beam_fl_t)
    if beam_fl_e is not None:
        g.add_beam(beam_ide, beam_fl_e)
    if beam_fl_b is not None:
        g.add_beam(beam_idb, beam_fl_b)

    mask_pol = 'm2' if mask_alm_p is not None else 'm1'

    ret_TT = g.decoupled_cl(alm_t, alm_t, 'm1',
                            spectype='TT', bin_weight_id='b1',
                            return_theory_filter=return_theory_filter,
                            beam_id1=beam_idt, beam_id2=beam_idt)
    ret_EE = g.decoupled_cl(alm_e, alm_e, mask_pol,
                            spectype='EE', bin_weight_id='b1',
                            return_theory_filter=return_theory_filter,
                            beam_id1=beam_ide, beam_id2=beam_ide)
    ret_BB = g.decoupled_cl(alm_b, alm_b, mask_pol,
                            spectype='BB', bin_weight_id='b1',
                            return_theory_filter=return_theory_filter,
                            beam_id1=beam_idb, beam_id2=beam_idb)
    ret_TE = g.decoupled_cl(alm_t, alm_e, 'm1', mask_pol,
                            spectype='TE', bin_weight_id='b1',
                            return_theory_filter=return_theory_filter,
                            beam_id1=beam_idt, beam_id2=beam_ide)
    return (*ret_TT, *ret_EE, *ret_BB, *ret_TE)


def alm2cross_power_spin2(lmax, alm_e1, alm_b1, alm_e2, alm_b2,
                          mask_alm1, mask_alm2=None,
                          beam_fl_e1=None, beam_fl_b1=None,
                          beam_fl_e2=None, beam_fl_b2=None,
                          bin_edges=None, bin_weights=None,
                          return_theory_filter=True):
    r"""
    Compute decoupled cross-power spectra (EE and BB) between two spin-2 fields 
    using masked sky observations, with support for independent beam deconvolution.

    This function estimates the cross angular power spectra between two spin-2 fields 
    given by their E/B-mode spherical harmonic coefficients. It corrects for mode 
    coupling due to partial sky coverage, supports optional beam deconvolution 
    for each field independently, and applies optional multipole binning. The result 
    is a debiased, decoupled cross-spectrum suitable for comparison with theoretical models.

    Parameters
    ----------
    lmax : int
        Maximum multipole (`ell`) up to which the power spectra are computed.

    alm_e1 : array_like
        E-mode spherical harmonic coefficients from the first spin-2 field.

    alm_b1 : array_like
        B-mode spherical harmonic coefficients from the first spin-2 field.

    alm_e2 : array_like
        E-mode spherical harmonic coefficients from the second spin-2 field.

    alm_b2 : array_like
        B-mode spherical harmonic coefficients from the second spin-2 field.

    mask_alm1 : array_like
        Spherical harmonic coefficients of the mask applied to the first field. 
        Should be defined up to `2 * lmax`.

    mask_alm2 : array_like or None, optional
        Spherical harmonic coefficients of the mask applied to the second field. 
        If `None`, `mask_alm1` is reused for both fields. Should be defined up to `2 * lmax`.

    beam_fl_e1 : array_like or None, optional
        Beam transfer function for the E-mode of the first field.

    beam_fl_b1 : array_like or None, optional
        Beam transfer function for the B-mode of the first field.

    beam_fl_e2 : array_like or None, optional
        Beam transfer function for the E-mode of the second field.

    beam_fl_b2 : array_like or None, optional
        Beam transfer function for the B-mode of the second field.

    bin_edges : array_like, optional
        Multipole bin edges for bandpower binning. If not provided, the output spectra 
        will not be binned. Note, these are of the form low_edge <= ℓ < upper_edge.

    bin_weights : array_like, optional
        Weights applied to each multipole within a bin. Useful for inverse-variance 
        or custom binning schemes.

    return_theory_filter : bool, default=True
        If `True`, also returns the theory bandpower filter matrices 
        :math:`\mathcal{F}^{++}_{q\ell}` and :math:`\mathcal{F}^{--}_{q\ell}` 
        for EE and BB, which can be used to convolve theoretical spectra 
        for comparison.

    Returns
    -------
    cl_EE : ndarray
        Decoupled EE cross-power spectrum as a 1D array over bins.

    tf_EE : ndarray or None
        Theory bandpower filter matrix for EE. Returned only if `return_theory_filter` is True.

    cl_BB : ndarray
        Decoupled BB cross-power spectrum as a 1D array over bins.

    tf_BB : ndarray or None
        Theory bandpower filter matrix for BB. Returned only if `return_theory_filter` is True.

    Notes
    -----
    - Uses the `Wiggle` class to apply masking, binning, beam deconvolution, 
      and decoupling.
    - Suitable for cross-correlations between different spin-2 fields such as 
      CMB polarization or cosmic shear measurements from different surveys.
    """

    g = Wiggle(lmax, bin_edges=bin_edges)
    g.add_mask('m1', mask_alm1)
    if mask_alm2 is not None:
        g.add_mask('m2', mask_alm2)
    g.add_bin_weights('b1', bin_weights=bin_weights)

    mask2 = 'm2' if mask_alm2 is not None else 'm1'

    beam_id_e1 = 'be1' if beam_fl_e1 is not None else None
    beam_id_b1 = 'bb1' if beam_fl_b1 is not None else None
    beam_id_e2 = 'be2' if beam_fl_e2 is not None else None
    beam_id_b2 = 'bb2' if beam_fl_b2 is not None else None

    if beam_fl_e1 is not None:
        g.add_beam(beam_id_e1, beam_fl_e1)
    if beam_fl_b1 is not None:
        g.add_beam(beam_id_b1, beam_fl_b1)
    if beam_fl_e2 is not None:
        g.add_beam(beam_id_e2, beam_fl_e2)
    if beam_fl_b2 is not None:
        g.add_beam(beam_id_b2, beam_fl_b2)

    ret_EE = g.decoupled_cl(alm_e1, alm_e2, 'm1', mask2, spectype='EE',
                            bin_weight_id='b1',
                            beam_id1=beam_id_e1, beam_id2=beam_id_e2,
                            return_theory_filter=return_theory_filter)

    ret_BB = g.decoupled_cl(alm_b1, alm_b2, 'm1', mask2, spectype='BB',
                            bin_weight_id='b1',
                            beam_id1=beam_id_b1, beam_id2=beam_id_b2,
                            return_theory_filter=return_theory_filter)

    return (*ret_EE, *ret_BB)

def alm2cross_power_spin02(lmax, alm_t1, alm_e1, alm_b1, alm_t2, alm_e2, alm_b2,
                           mask_alm_t1, mask_alm_t2=None,
                           mask_alm_p1=None, mask_alm_p2=None,
                           bin_edges=None, bin_weights=None,
                           beam_fl_t1=None, beam_fl_e1=None, beam_fl_b1=None,
                           beam_fl_t2=None, beam_fl_e2=None, beam_fl_b2=None,
                           return_theory_filter=True):
    r"""
    Compute decoupled cross-power spectra (TT, EE, BB, TE) between masked fields consisting of 
    a scalar (spin-0) component and a spin-2 polarization component.

    This function estimates cross angular power spectra between two datasets defined by 
    scalar and spin-2 spherical harmonic coefficients. It corrects for the effects of sky masking, 
    optionally applies beam deconvolution independently for each field and component, 
    and performs multipole binning. The output is suitable for unbiased comparison with 
    theoretical models in cosmological analyses (e.g., CMB temperature and polarization, 
    galaxy clustering and shear).

    Parameters
    ----------
    lmax : int
        Maximum multipole (`ell`) up to which the power spectra are computed.

    alm_t1 : array_like
        Spherical harmonic coefficients of the scalar (T) map from the first field.

    alm_e1 : array_like
        Spherical harmonic coefficients of the E-mode polarization map from the first field.

    alm_b1 : array_like
        Spherical harmonic coefficients of the B-mode polarization map from the first field.

    alm_t2 : array_like
        Spherical harmonic coefficients of the scalar (T) map from the second field.

    alm_e2 : array_like
        Spherical harmonic coefficients of the E-mode polarization map from the second field.

    alm_b2 : array_like
        Spherical harmonic coefficients of the B-mode polarization map from the second field.

    mask_alm_t1 : array_like
        Mask (in harmonic space) applied to the scalar field of the first dataset. 
        Should be defined up to `2 * lmax`.

    mask_alm_t2 : array_like or None, optional
        Mask applied to the scalar field of the second dataset. If `None`, `mask_alm_t1` is reused.

    mask_alm_p1 : array_like or None, optional
        Mask applied to the spin-2 fields (E and B) of the first dataset. 
        If `None`, `mask_alm_t1` is reused.

    mask_alm_p2 : array_like or None, optional
        Mask applied to the spin-2 fields of the second dataset. 
        If `None`, `mask_alm_p1` is reused.

    bin_edges : array_like, optional
        Multipole bin edges for bandpower binning. If not provided, the output spectra are unbinned.
        Note, these are of the form low_edge <= ℓ < upper_edge.

    bin_weights : array_like, optional
        Weights applied to each multipole within a bin.

    beam_fl_t1 : array_like or None, optional
        Beam transfer function for the scalar field of the first dataset. If None, no beam is deconvolved.

    beam_fl_e1 : array_like or None, optional
        Beam transfer function for the E-mode map of the first dataset. If None, no beam is deconvolved.

    beam_fl_b1 : array_like or None, optional
        Beam transfer function for the B-mode map of the first dataset. If None, no beam is deconvolved.

    beam_fl_t2 : array_like or None, optional
        Beam transfer function for the scalar field of the second dataset. If None, no beam is deconvolved.

    beam_fl_e2 : array_like or None, optional
        Beam transfer function for the E-mode map of the second dataset. If None, no beam is deconvolved.

    beam_fl_b2 : array_like or None, optional
        Beam transfer function for the B-mode map of the second dataset. If None, no beam is deconvolved.

    return_theory_filter : bool, default=True
        If `True`, also returns the theory bandpower filter matrices 
        :math:`\mathcal{F}^{s_as_b}_{q\ell}` for TT, EE, BB, and TE, which can be used 
        to convolve theoretical spectra for comparison.

    Returns
    -------
    cl_TT : ndarray
        Decoupled TT cross-power spectrum.

    tf_TT : ndarray or None
        Theory bandpower filter matrix for TT. Returned only if `return_theory_filter` is True.

    cl_EE : ndarray
        Decoupled EE cross-power spectrum.

    tf_EE : ndarray or None
        Theory bandpower filter matrix for EE. Returned only if `return_theory_filter` is True.

    cl_BB : ndarray
        Decoupled BB cross-power spectrum.

    tf_BB : ndarray or None
        Theory bandpower filter matrix for BB. Returned only if `return_theory_filter` is True.

    cl_T1E2 : ndarray
        Decoupled TE cross-power spectrum (T from first field, E from second).

    tf_T1E2 : ndarray or None
        Theory bandpower filter matrix for T1E2. Returned only if `return_theory_filter` is True.

    cl_T2E1 : ndarray
        Decoupled TE cross-power spectrum (T from second field, E from first).

    tf_T2E1 : ndarray or None
        Theory bandpower filter matrix for T2E1. Returned only if `return_theory_filter` is True.
    """

    g = Wiggle(lmax, bin_edges=bin_edges)
    g.add_mask('mt1', mask_alm_t1)
    if mask_alm_t2 is not None:
        g.add_mask('mt2', mask_alm_t2)
    t_mask2 = 'mt2' if mask_alm_t2 is not None else 'mt1'

    g.add_mask('mp1', mask_alm_p1 if mask_alm_p1 is not None else mask_alm_t1)
    if mask_alm_p2 is not None:
        g.add_mask('mp2', mask_alm_p2)
    p_mask2 = 'mp2' if mask_alm_p2 is not None else 'mp1'

    g.add_bin_weights('b1', bin_weights=bin_weights)

    beam_idt1 = beam_idt2 = beam_ide1 = beam_ide2 = beam_idb1 = beam_idb2 = None

    if beam_fl_t1 is not None:
        g.add_beam('pt1', beam_fl_t1)
        beam_idt1 = 'pt1'
    if beam_fl_t2 is not None:
        g.add_beam('pt2', beam_fl_t2)
        beam_idt2 = 'pt2'
    if beam_fl_e1 is not None:
        g.add_beam('pe1', beam_fl_e1)
        beam_ide1 = 'pe1'
    if beam_fl_e2 is not None:
        g.add_beam('pe2', beam_fl_e2)
        beam_ide2 = 'pe2'
    if beam_fl_b1 is not None:
        g.add_beam('pb1', beam_fl_b1)
        beam_idb1 = 'pb1'
    if beam_fl_b2 is not None:
        g.add_beam('pb2', beam_fl_b2)
        beam_idb2 = 'pb2'

    ret_TT = g.decoupled_cl(alm_t1, alm_t2, 'mt1', t_mask2, spectype='TT',
                            bin_weight_id='b1',
                            return_theory_filter=return_theory_filter,
                            beam_id1=beam_idt1, beam_id2=beam_idt2)
    ret_EE = g.decoupled_cl(alm_e1, alm_e2, 'mp1', p_mask2, spectype='EE',
                            bin_weight_id='b1',
                            return_theory_filter=return_theory_filter,
                            beam_id1=beam_ide1, beam_id2=beam_ide2)
    ret_BB = g.decoupled_cl(alm_b1, alm_b2, 'mp1', p_mask2, spectype='BB',
                            bin_weight_id='b1',
                            return_theory_filter=return_theory_filter,
                            beam_id1=beam_idb1, beam_id2=beam_idb2)
    ret_T1E2 = g.decoupled_cl(alm_t1, alm_e2, 'mt1', p_mask2, spectype='TE',
                            bin_weight_id='b1',
                            return_theory_filter=return_theory_filter,
                            beam_id1=beam_idt1, beam_id2=beam_ide2)
    ret_T2E1 = g.decoupled_cl(alm_t2, alm_e1, t_mask2, 'mp1', spectype='TE',
                            bin_weight_id='b1',
                            return_theory_filter=return_theory_filter,
                            beam_id1=beam_idt2, beam_id2=beam_ide1)

    return (*ret_TT, *ret_EE, *ret_BB, *retT1E2, *ret_T2E1)
