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

import os

_reserved_bin_id = '__unity'


"""
=========
Core classes and functions
=========
"""


class Wiggle(object):
    """
    Position-space Power Spectrum Mode-Decoupler
    =============================================

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

    def _zeros(self,):
        if self._binned:
            return np.zeros((self.nbins,self.nbins),dtype=np.float64)
        else:
            return np.zeros((self.lmax+1,self.lmax+1),dtype=np.float64)
        
    def _get_corr(self,mask_id1,mask_id2,parity):
        if mask_id2 is None: mask_id2 = mask_id1
        # this gives me a parity weighted correlation function from mask Cls
        coeff = (2*self.ells+1)/(4*np.pi) * self._get_mask_cls(mask_id1,mask_id2=mask_id2)[:(self.xlmax*self.lmax+1)] # (2*lmax+1,)
        coeff = _parity_flip(coeff,parity) # this applies (-1)^ell if parity is '-'
        # cd00 are just Legendre polynomials (or Wigner d_00)
        xi = self.cd00 @ coeff # (N,)
        return xi

    def _get_G_term_weights(self,mode_count_weight,parity,apply_bin_weights,bin_weight_id,pfact=None):
        # Weights needed for the the things that multiply Wigner-ds in the G-matrices
        # Start building weights

        # Unity bin weights if none specified
        if self._binned and (bin_weight_id is None):
            self._populate_unity_bins()
            bin_weight_id = _reserved_bin_id
        
        ws = np.ones(self.lmax+1)
        ws = _parity_flip(ws,parity)  # this applies (-1)^ell if parity is '-'
        if mode_count_weight:
            ws = ws * ((2*self.ells[:self.lmax+1]+1)/2.)
        if self._binned and apply_bin_weights:
            ws = ws * self._nweights[str(bin_weight_id)]
        # For purification
        if pfact is not None:
            ls = self.ells[:self.lmax+1]
            mul = np.sqrt((ls-1)*ls*(ls+1)*(ls+2))
            ws[ls>=2] = ws[ls>=2] * (mul[ls>=2])**pfact
        return ws

    def _get_b1_b2(self,spin1,spin2,parity,bin_weight_id,
                   beam_id1,beam_id2,gfact=None):
        # Get the left (ell) and right (ell') sides of the G-matrices
        
        # these are weights that multiply the ell side of the G matrix
        # it includes a parity term as well as bandpower bin weights
        nweights = self._get_G_term_weights(mode_count_weight=False,parity=parity,apply_bin_weights=True,bin_weight_id=bin_weight_id,
                                            pfact=(-gfact) if (gfact is not None) else None)

        
        # these are weights that multiply the ell' side of the G matrix
        # it includes a parity term and a mode counting term but no bin weights
        nweights2 = self._get_G_term_weights(mode_count_weight=True,parity=parity,apply_bin_weights=False,bin_weight_id=None,
                                             pfact=gfact if (gfact is not None) else None)
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

    def _get_wigner(self,spin1,spin2):
        return _wiggle._compute_wigner_d_matrix(self.lmax,spin1,spin2,self.mu)
        
    def _get_m(self,mask_id1,mask_id2,spin1,spin2,parity,bin_weight_id,
               beam_id1,beam_id2,gfact=None):
        # Get the core of the mode-coupling G matrices. This is where the expensive
        # dot product happens.

        # Know of situations where other spins would be useful? Write to us!
        if [spin1,spin2] not in [[0,0],[2,2],[2,0]]: raise NotImplementedError

        # Get the left (ell) and right (ell') sides of the G-matrices
        b1,b2 = self._get_b1_b2(spin1,spin2,parity,bin_weight_id=bin_weight_id,beam_id1=beam_id1,beam_id2=beam_id2,gfact=gfact)
            
        # Get the Gauss-Legendre quadrature weighted correlation functions of the mask
        xi = self._get_corr(mask_id1,mask_id2,parity=parity)
        W = self.w_mu * xi
        
        # Know how to speed up matrix products? Write to us! (This one does use OpenMP threads)
        M = np.einsum('i,ij,ik->jk', W, b1, b2, optimize='greedy')
        return M
    
    def get_coupling_matrix_from_ids(self,mask_id1,mask_id2,spintype,bin_weight_id=None,
                                          beam_id1=None,beam_id2=None, pure_E=False,pure_B=False):
        """
        Compute the mode-coupling matrix for a given pair of masks and spin configuration.

        This method retrieves the mode-coupling matrix used in power spectrum estimation, 
        based on the specified spin configuration (`spintype`) and mask identifiers. 
        The method supports scalar (spin-0) and spin-2 field combinations, e.g. for use 
        in CMB or large-scale structure analyses.

        Parameters
        ----------
        mask_id1 : str or int
            Identifier for the first mask to use in the coupling matrix calculation.
        mask_id2 : str or int
            Identifier for the second mask to use in the coupling matrix calculation.
        spintype : str
            The spin configuration to compute the coupling matrix for. Supported values are:
                - '00' : scalar-scalar (spin-0 × spin-0)
                - '++' : spin-2 auto (E-mode like)
                - '--' : spin-2 cross (B-mode like)
                - '20' : spin-2 × spin-0 mixed term
        bin_weight_id : str or int, optional
            Identifier for the binning weights, if applicable.
        beam_id1 : str or int, optional
            Identifier for the beam function in the first map.
        beam_id2 : str or int, optional
            Identifier for the beam function in the second map.

        Returns
        -------
        numpy.ndarray
            The computed coupling matrix corresponding to the specified spin type and masks.

        Raises
        ------
        ValueError
            If an unsupported `spintype` is provided.
        """
        
        if spintype not in ['00','20_E','20_B','22','+','-']: raise ValueError(f'spintype {spintype} not recognized')
        f = lambda spin1,spin2,parity,gfact: self._get_m(mask_id1,mask_id2,spin1=spin1,spin2=spin2,parity=parity,
                                                    bin_weight_id=bin_weight_id,
                                                    beam_id1=beam_id1,beam_id2=beam_id2,gfact=gfact)

        return self._Mmatrix(spintype,f,pure_E,pure_B)

    def _Mmatrix(self,spintype,f,pure_E,pure_B):
        def Mp(npure):
            if npure==0:
                g1 = f(2,2,'+',None)
                g2 = f(2,2,'-',None)
            elif npure==1:
                g1 = f(2,0,'+',1)
                g2 = f(2,0,'-',1)
            elif npure==2:
                g1 = f(0,0,'+',2)
                g2 = f(0,0,'-',2)                
            return (g1+g2)/2.

        if spintype=='00':
            return f(0,0,'+',None)
        elif spintype[:2]=='20':
            if (('E' in spintype) and pure_E) or (('B' in spintype) and pure_B):
                return f(0,0,'+',1)
            else:
                return f(2,0,'+',None)
        elif spintype in ['+','-']:
            # These are just used in tests
            if pure_E or pure_B: raise ValueError
            g1 = f(2,2,'+',None)
            g2 = f(2,2,'-',None)
            if spintype=='+':
                return (g1+g2)/2.
            elif spintype=='-':
                return (g1-g2)/2.
        elif spintype=='22':

            zero = self._zeros()
            
            if pure_E and pure_B:
                Mm_EB = zero # TODO: confirm this identity
            else:
                if any([pure_E,pure_B]):
                    g1 = f(2,0,'+',1)
                    g2 = f(2,0,'-',1)
                else:
                    g1 = f(2,2,'+',None)
                    g2 = f(2,2,'-',None)
                Mm_EB = (g1-g2)/2.

                
            Mp_EE = Mp(int(pure_E)*2)
            Mp_BB = Mp(int(pure_B)*2)
            Mp_EB = Mp(sum([pure_E,pure_B]))
            
            return np.block([
                [ Mp_EE,     zero,   zero,   Mm_EB  ],
                [ zero,   Mp_EB,    -Mm_EB,     zero],
                [ zero,  -Mm_EB,     Mp_EB,     zero],
                [ Mm_EB,     zero,   zero,   Mp_BB  ]
            ])
        

    def _get_mask_cls(self,mask_id1,mask_id2):
        if mask_id2 is None: mask_id2 = mask_id1
        try:
            mcls = self._mask_cl_cache[f'{mask_id1}_{mask_id2}']
            if self.verbose: print(f"Reusing mask cls {mask_id1}_{mask_id2}...")
            return mcls
        except KeyError:
            self._mask_cl_cache[f'{mask_id1}_{mask_id2}'] = alm2cl(self._mask_alm_cache[mask_id1],self._mask_alm_cache[mask_id2])
            return self._mask_cl_cache[f'{mask_id1}_{mask_id2}']

    def _thfilt_core(self,mask_id1,mask_id2,spin1,spin2,parity,bin_weight_id,beam_id1,beam_id2,gfact):
        if spin1==0 and spin2==0:
            ud = self.ud00
        elif spin1==2 and spin2==0:
            ud = self._get_wigner(2,0)
        elif spin1==2 and spin2==2:
            ud = self._get_wigner(2,2)
        b1,b2 = self._get_b1_b2(spin1,spin2,parity,bin_weight_id=bin_weight_id,beam_id1=beam_id1,beam_id2=beam_id2,gfact=gfact) # (N x nbins)
        xi = self._get_corr(mask_id1,mask_id2,parity=parity)
        W = self.w_mu * xi # (N,)
        b2w = self._get_G_term_weights(mode_count_weight=True,parity=parity,apply_bin_weights=False,bin_weight_id=None) # (nells,)
        R2 = ud * b2w[None,:] # (N, nells)
        M = np.einsum('i,ij,ik->jk', W, b1, R2, optimize='greedy') # (nbins,nells)
        return M
        
        
    def get_theory_filter(self,mask_id1,mask_id2=None,spintype='00',bin_weight_id=None,pure_E=False,pure_B=False):
        r"""
        Construct the theoretical bandpower filter :math:`\mathcal{F}^{s_as_b}_{q\ell}`,
        as defined in arxiv:1809.09603.

        This method computes the filter matrix that transforms the theoretical full-sky power 
        spectrum :math:`\mathsf{C}^{ab,\mathrm{th}}_\ell` into the corresponding prediction 
        for the *binned, decoupled* bandpowers :math:`\mathsf{B}^{ab,\mathrm{th}}_q` in the 
        presence of mode coupling and nontrivial binning. The filter is given by (see arxiv:1809.09603) :

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

        The theory spectrum this filter is dotted with must *not* contain the beam, even if beams were
        deconvolved during the Wiggle analysis.


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
        
        if not(self._binned): raise ValueError("No bin edges were specified when initializing Wiggle, so no theory filter can be determined.")
        if mask_id2 is None: mask_id2 = mask_id1
        f = lambda spin1, spin2, parity, gfact: self._thfilt_core(mask_id1,mask_id2,spin1,spin2,parity,bin_weight_id=bin_weight_id,
                                                             beam_id1=None,beam_id2=None, gfact=gfact)

        Mc = self._Mmatrix(spintype,f,pure_E,pure_B)
            
        cinv = self._get_cinv(mask_id1,mask_id2=mask_id2,spintype=spintype,bin_weight_id=bin_weight_id,
                              beam_id1=None,beam_id2=None)
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


    def _get_cinv(self,mask_id1,mask_id2,spintype,bin_weight_id,
                  beam_id1,beam_id2, pure_E=False,pure_B=False):
        mcm = self.get_coupling_matrix_from_ids(mask_id1,mask_id2,spintype=spintype,
                                                 beam_id1=beam_id1,beam_id2=beam_id2,bin_weight_id=bin_weight_id, pure_E=pure_E,pure_B=pure_B)
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

    def _bin_if_needed(self,cls,bin_weight_id):
        cls = cls[:self.lmax+1]
        # Bin Cls
        if self._binned:
            return bin_array(cls, self._bin_indices, self.nbins,weights=self._nweights[str(bin_weight_id)])
        else:
            return cls        

    def get_powers(self,alms1,alms2, mask_ids1, mask_ids2=None,
                     bin_weight_id = None,
                     beam_id1 = None, beam_id2 = None, pure_E = False, pure_B = False,
                     return_theory_filter=False):

        r"""
        Compute decoupled angular power spectra (:math:`C_{\ell}`) from the spherical harmonics
        of maps that have already been masked.

        This method estimates the angular power spectrum between two input fields 
        in harmonic space (`alm`s), accounting for mode coupling due to partial sky coverage,
        beam smoothing, and bandpower binning. The result is a debiased, decoupled bandpower 
        estimate suitable for direct comparison with theoretical predictions. Note that
        a theory filter of shape (nbins,nells) needs to be applied to (nells,) shaped
        theory spectra (no beam) if using bandpower binning. This filter can be obtained from this
        function call, but is the most expensive part of the calculation, so consider
        obtaining it from `self.get_theory_filter` once.

        Parameters
        ----------
        alms1, alms2 : ndarray
            Spherical-harmonic coefficients of the first and second map,
            both with shape

            * ``(N_alm,)``           – a single scalar field;
            * ``(1, N_alm)``         – same as above (explicit 1-pol);
            * ``(2, N_alm)``         – spin-2 field (*E*, *B*);
            * ``(3, N_alm)``         – scalar + spin-2 (*T*, *E*, *B*).

        The two arrays **must be identical in shape**.  If you pass the
        *same* object twice the function returns auto-spectra.
        

        mask_ids1 : str or sequence(str)
            Mask identifier(s) previously registered via :py:meth:`add_mask`
            for the first field.  If you pass a single string it is applied
            to all polarisation components; otherwise give a two-element list/tuple with
            one entry for the spin-0 field and the second for the spin-2 field.

        mask_ids2 : str or sequence(str), optional
            Mask identifier(s) for the second field.  If *None* (default) the
            same mask(s) as *mask_ids1* are used.

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

        ----------
        Returns
        -------
        spectra : dict
            Keys and shapes depend on the input polarisation:

            * scalar–scalar -> ``{'TT': {'Cls': (nbin | lmax+1,)}}``
            * spin-2 only   -> ``{'EE', 'EB', 'BE', 'BB'}``
            * scalar+spin-2 -> ``{'TT', 'TE', 'ET', 'TB', 'BT',
                                  'EE', 'EB', 'BE', 'BB'}``

            Each entry holds a dict with the field ``'Cls'`` containing the
            decoupled (and possibly binned) spectrum.

            If *return_theory_filter* is *True*, a second key ``'Th'`` is
            added containing the corresponding theory filters with 
            shape ``(n_bins, lmax+1)``.

            For the pure-polarization part, the theory filter combines
            EE, EB, BE and BB and is in spectra['ThPol']. For most cases
            involving null EB, BE and BB, you only need the EE portion of this
            matrix.

        ----------
        Raises
        ------
        ValueError
            * The two ``alm`` arrays differ in shape.
            * Unsupported number of polarisation components.
            * Inconsistent mask list length.
            * Internal binning/mixing dimensions do not match.
        

        Notes
        -----
        - This method is the recommended interface for evaluating both auto- and 
          cross-spectra in masked sky analyses, but other convenience wrappers are provided elsewhere.

        """

        if alms1.shape != alms2.shape: raise ValueError
        if alms1.ndim==1:
            alms1 = alms1[None,:]
            alms2 = alms2[None,:]
        else:
            if alms1.ndim>2 or alms1.ndim<1: raise ValueError
        npol = alms1.shape[0]
        idoff = 0
        if npol==1:
            spin = "T"
        elif npol==2:
            spin = "P"
            idoff = -1
        elif npol==3:
            spin = "T+P"
        else:
            raise ValueError
            
        if mask_ids2 is None: mask_ids2 = mask_ids1
        if isinstance(mask_ids1,str): mask_ids1 = [mask_ids1]*2
        if isinstance(mask_ids2,str): mask_ids2 = [mask_ids2]*2
        
        # Unity bin weights if none specified
        if self._binned and (bin_weight_id is None):
            self._populate_unity_bins()
            bin_weight_id = _reserved_bin_id

        balm2cl = lambda a1,a2: self._bin_if_needed(alm2cl(a1,a2),bin_weight_id)
        decfunc = lambda cls,spintype,m1,m2: self._decouple_binned_cl(cls, m1, mask_id2=m2, spintype=spintype,
                                                          bin_weight_id = bin_weight_id,
                                                          beam_id1 = beam_id1, beam_id2 = beam_id2,
                                                          return_theory_filter=return_theory_filter, pure_E=pure_E,pure_B=pure_B)
        
        ret_cls = {}
        if 'T' in spin:
            # Get TT
            cls_tt = balm2cl(alms1[0], alms2[0])
            ret_cls['TT'] = decfunc(cls_tt,'00',mask_ids1[0],mask_ids2[0])
        if '+' in spin:
            # Get TE
            cls_te = balm2cl(alms1[0], alms2[1])
            ret_cls['TE'] = decfunc(cls_te,'20_E',mask_ids1[0],mask_ids2[1])
            cls_et = balm2cl(alms1[1], alms2[0])
            ret_cls['ET'] = decfunc(cls_et,'20_E',mask_ids1[1],mask_ids2[0])
            # Get TB
            cls_tb = balm2cl(alms1[0], alms2[2])
            ret_cls['TB'] = decfunc(cls_tb,'20_B',mask_ids1[0],mask_ids2[1])
            cls_bt = balm2cl(alms1[2], alms2[0])
            ret_cls['BT'] = decfunc(cls_bt,'20_B',mask_ids1[1],mask_ids2[0])
        if 'P' in spin:
            # Get EE
            cls_ee = balm2cl(alms1[1+idoff], alms2[1+idoff])
            # Get EB
            cls_eb = balm2cl(alms1[1+idoff], alms2[2+idoff])
            # Get EB
            cls_be = balm2cl(alms1[2+idoff], alms2[1+idoff])
            # Get BB
            cls_bb = balm2cl(alms1[2+idoff], alms2[2+idoff])

            cls_pol = np.concatenate([cls_ee, cls_eb,cls_be, cls_bb])
            ret = decfunc(cls_pol,'22',mask_ids1[1],mask_ids2[1])

            if return_theory_filter: ret_cls['ThPol'] = ret['Th']
            # Unpack concatenated decoupled spectra
            rlist = ['EE','EB','BE','BB']
            start = 0
            if self._binned:
                step = self.nbins
            else:
                step = self.lmax+1
            for i,spec in enumerate(rlist):
                end = start + step
                ret_cls[spec] = {'Cls':ret['Cls'][start:end]}
                if ret_cls[spec]['Cls'].size!=step: raise ValueError
                start = end
                
        return ret_cls
            
            

    def decoupled_cl(self,pcls, mask_id1, mask_id2=None, spintype='00',
                           bin_weight_id = None,
                           beam_id1 = None, beam_id2 = None,
                           return_theory_filter=False, pure_E=False,pure_B=False):
        # pcls must not be binned. This function is not suitable for spin!=0.
            
        # Unity bin weights if none specified
        if self._binned and (bin_weight_id is None):
            self._populate_unity_bins()
            bin_weight_id = _reserved_bin_id
            
        pcls = self._bin_if_needed(pcls,bin_weight_id)
        return self._decouple_binned_cl(pcls, mask_id1, mask_id2, spintype,
                           bin_weight_id,
                           beam_id1, beam_id2,
                           return_theory_filter, pure_E,pure_B)
        
    def _decouple_binned_cl(self,pcls, mask_id1, mask_id2=None, spintype='00',
                           bin_weight_id = None,
                           beam_id1 = None, beam_id2 = None,
                           return_theory_filter=False, pure_E=False,pure_B=False):
        # pcls must already be binned. It can be a concatenation of polarization spectra.

        if spintype not in ['00','20_B','20_E','22']: raise NotImplementedError

        # Get MCM
        cinv = self._get_cinv(mask_id1,mask_id2=mask_id2,spintype=spintype,
                              bin_weight_id=bin_weight_id,
                              beam_id1=beam_id1,beam_id2=beam_id2, pure_E=pure_E,pure_B=pure_B)
        # Decouple
        dcls = np.dot(cinv,pcls)

        ret = {}
        ret['Cls'] = dcls
        if return_theory_filter:
            ret['Th'] = self.get_theory_filter(mask_id1,mask_id2,spintype=spintype,
                                               bin_weight_id=bin_weight_id,
                                               pure_E=pure_E,pure_B=pure_B)
        return ret
    

def get_coupling_matrix_from_mask_cls(mask_cls,lmax,spintype='00',bin_edges = None,bin_weights = None,
                                      beam_fl1 = None,beam_fl2 = None, pure_E=False,pure_B=False,
                                      return_obj=False):
    r"""
    Compute the  (optionally, binned) mode-coupling matrix from the pseudo-Cl of a sky mask.

    This function is a high-level wrapper to generate the  (optionally, binned) mode-coupling matrix 
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
        The  (optionally, binned) mode-coupling matrix of shape `(n_bins, n_bins)` if binned or `(lmax+1, lmax+1)` if unbinned.

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
    m = g.get_coupling_matrix_from_ids('m1','m1',spintype=spintype,bin_weight_id='b1',
                                       beam_id1=beam_id1,beam_id2=beam_id2, pure_E=pure_E,pure_B=pure_B)
    if return_obj:
        return m,g
    return m



def get_pure_EB_alms(Qmap, Umap, mask, masked_on_input=False,
           return_mask=False, lmax=None,
           eps=1e-4):

    """
    Perform pure E/B mode decomposition on a masked polarization map. Adapted
    from code by Will Coulton and Irene Abril-Cabeszas.

    Implements the Smith & Zaldarriaga (2007) formalism to compute 
    "pure" E and B-mode multipoles from Q/U Stokes parameter maps 
    using an apodized mask, removing ambiguous modes due to masking.

    Parameters
    ----------
    Q : ndarray or enmap
        Stokes Q polarization map. Should be masked if `masked_on_input=True`.
    U : ndarray or enmap
        Stokes U polarization map. Same shape and type as Q.
    mask : ndarray
        Scalar apodized mask, typically smoothed to avoid ringing.
        Should be in the same geometry as Q and U (Healpix or enmap).
    masked_on_input : bool, optional
        If True, assumes Q and U have already been multiplied by mask.
        If False, the code will divide out the mask (with threshold protection).
        Default is True.
    return_mask : bool, optional
        If True, returns the spin-1 and spin-2 derivatives of the mask
        instead of the pure E/B multipoles. Useful for debugging.
        Default is False.
    lmax : int or None, optional
        Maximum multipole to compute. If None, defaults to half the Nyquist
        limit of the map.
    tiny : float, optional
        Threshold below which mask values are treated as zero to avoid
        division by very small values. Default is 1e-4.

    Returns
    -------
    pureE_alms : ndarray
        Spherical harmonic coefficients of the purified E-mode signal,
        shape consistent with `cs.alm_info(lmax)`.
    pureB_alms : ndarray
        Spherical harmonic coefficients of the purified B-mode signal.
    
    OR

    mask_1, mask_2 : ndarray
        If `return_mask=True`, returns the spin-1 and spin-2 derivatives 
        of the mask (used in the purification process) instead of pureE/B.

    Notes
    -----
    - Requires the `pixell` library for spherical harmonic transforms and 
      map handling.
    - This method suppresses E->B leakage caused by the mask, and
      is critical for detecting weak B-mode signals like those from 
      primordial gravitational waves.
    - Based on the formalism in:
        * Smith, Kendrick M. "Pseudo-Cl estimators which do not mix 
          E and B modes." Phys. Rev. D 74.8 (2006): 083002.
        * Smith & Zaldarriaga, Phys. Rev. D 76.4 (2007): 043001.

    Examples
    --------
    >>> pureE_alms, pureB_alms = get_pure_EB_alms(Qmap, Umap, apod_mask, is_healpix=True)
    
    >>> mask_1, mask_2 = get_pure_EB_alms(Qmap, Umap, apod_mask, return_mask=True)
    """
    from pixell import enmap, curvedsky as cs



    def spins(alm2map, alm, mp12, spin):
        if spin == 0:
            assert(0)
        SP, SM = alm2map(np.array([-alm, alm*0.]), mp12, spin=abs(spin))
        return SP + 1j*SM if spin > 0 else SP - 1j*SM

    # Un-mask input maps if needed
    if masked_on_input:
        good = mask > eps
        Q = enmap.enmap(np.where(good, Qmap / mask, 0.0), Qmap.wcs)
        U = enmap.enmap(np.where(good, Umap / mask, 0.0), Umap.wcs)
    else:
        Q = Qmap
        U = Umap

    slmax = int(np.min(np.pi / Q.pixshape() / 2))
    if lmax is None:
        lmax = slmax
    else:
        if lmax>slmax: warnings.warn(f"Your pixelization supports lmax={slmax} but you have requested {lmax}. You will likely have excess power on large scales! E/B purification requires accurate SHTs, so an lmax of pi/pix_size/2 is recommended.")
    map2alm, alm2map = cs.map2alm, cs.alm2map
    template = enmap.zeros((2,) + Q.shape, wcs=Q.wcs,dtype=np.float64)

    ainfo = cs.alm_info(lmax)
    ells  = np.arange(lmax+1, dtype=np.float64)

    # Mask derivatives
    wAlm = map2alm(mask, ainfo=ainfo, spin=0)

    fac1 = np.zeros_like(ells)
    fac1[1:] = np.sqrt(ells[1:] * (ells[1:] + 1))
    fac2 = np.zeros_like(ells)
    fac2[2:] = np.sqrt((ells[2:] - 1)*ells[2:]*(ells[2:] + 1)*(ells[2:] + 2))
                                                 

    wAlm1 = ainfo.lmul(wAlm.copy(), fac1)        
    wAlm2 = ainfo.lmul(wAlm.copy(), fac2)        

    mask1 = spins(alm2map, wAlm1, template*0, 1) 
    mask2 = spins(alm2map, wAlm2, template*0, 2) 
    mask1[mask < eps] = 0.0                    
    mask2[mask < eps] = 0.0
    if return_mask:
        return mask1, mask2

    template[...] = (Q*mask, U*mask)
    E2, B2 = map2alm(template, ainfo=ainfo, spin=2)

    template[...] = (Q*mask1.real + U*mask1.imag,
                     U*mask1.real - Q*mask1.imag)
    E1, B1 = map2alm(template, ainfo=ainfo, spin=1)

    E0 = map2alm(Q*mask2.real + U*mask2.imag, ainfo=ainfo, spin=0)
    B0 = map2alm(U*mask2.real - Q*mask2.imag, ainfo=ainfo, spin=0)

    alpha = np.zeros_like(ells)
    alpha[2:] = 2.0 * np.sqrt(1.0 / ((ells[2:] + 2)*(ells[2:] - 1)))
    beta  = np.zeros_like(ells)
    beta[2:]  = np.sqrt(1.0 / ((ells[2:] + 2)*(ells[2:] + 1)*ells[2:]*(ells[2:] - 1)))

    pureE_alms = E2 + ainfo.lmul(E1, alpha) - ainfo.lmul(E0, beta)
    pureB_alms = B2 + ainfo.lmul(B1, alpha) - ainfo.lmul(B0, beta)

    # Strip monopole / dipole
    killMD = np.ones_like(ells); killMD[:2] = 0.0
    pureE_alms  = ainfo.lmul(pureE_alms, killMD)
    pureB_alms  = ainfo.lmul(pureB_alms, killMD)

    return pureE_alms, pureB_alms


