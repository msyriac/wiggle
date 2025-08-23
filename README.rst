wiggle
======

.. image:: https://img.shields.io/github/actions/workflow/status/msyriac/wiggle/build.yml?branch=main
   :target: https://github.com/msyriac/wiggle/actions?query=workflow%3ABuild
   :alt: Github Actions Status
.. image:: https://readthedocs.org/projects/wiggle/badge/?version=latest
   :target: https://wiggle.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/pypi/v/pywiggle.svg
   :target: https://pypi.org/project/pywiggle/
   :alt: PyPI version

\
\

.. image:: https://raw.githubusercontent.com/msyriac/wiggle/main/docs/wiggle_logo.png
   :width: 150
   :alt: Wiggle logo
	
``wiggle`` stands for the WIGner Gauss-Legendre Estimator. This Python package provides a fast implementation of unbiased angular power spectrum estimation of spin-0 and spin-2 fields on the sphere, most commonly encountered in the context of cosmological data analysis. With an efficient and exact algorithm, this code performs mode-decoupling very fast; in the case of binned spectra, ``wiggle`` can be orders of magnitude faster than other approaches (often around one second of compute-time at most).


⚠️  **Warning**: ``wiggle`` is under active development! Its API will likely change.


* Free software: BSD license
* Documentation: https://wiggle.readthedocs.io.


  
Installing
----------

Make sure your ``pip`` tool is up-to-date. To install ``wiggle``, run:

.. code-block:: console
		
   $ pip install pywiggle --user

This will install a pre-compiled binary suitable for your system (only Linux and Mac OS X with Python>=3.9 are supported). After installation, make sure to run a test with:

.. code-block:: console
		
   $ pytest --pyargs pywiggle.tests

If you require more control over your installation, e.g. using Intel compilers, please see the section below on compiling from source.

Compiling from source (advanced / development workflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install from source is to use the ``pip`` tool,
with the ``--no-binary`` flag. This will download the source distribution
and compile it for you. Don't forget to make sure you have CXX set
if you have any problems.

For all other cases, below are general instructions.

First, download the source distribution or ``git clone`` this repository. You
can work from ``master`` or checkout one of the released version tags (see the
Releases section on Github). Then change into the cloned/source directory.

Once downloaded, you can install using ``pip install .`` inside the project
directory. We use the ``meson`` build system, which should be understood by
``pip`` (it will build in an isolated environment).

We suggest you then test the installation by running the unit tests. You
can do this by running ``pytest``.

To run an editable install, you will need to do so in a way that does not
have build isolation (as the backend build system, `meson` and `ninja`, actually
perform micro-builds on usage in this case):

.. code-block:: console
   
   $ pip install --upgrade pip meson ninja meson-python cython numpy pybind11
   $ pip install  --no-build-isolation --editable .

After installation, make sure to run a test with:
   
.. code-block:: console
   
   $ pytest

Quick Usage
-----------

Accurate power spectrum estimation requires you to first convert a pixelated and masked map to its spherical harmonic coefficients. ``wiggle`` does not provide tools for SHTs and expects you to have the ``alm`` coefficients both for the masked fields and the mask itself already in hand.  These can be obtained using a code like ``healpy`` in the case of HEALPix maps or a code like ``pixell`` in the case of rectangular pixelization maps.

If you are interested in accurate power spectra out to some maximum multipole ``lmax``, we recommend you evaluate SHTs out to ``lmax`` for the masked fields, but out to  ``2 lmax`` for the mask itself. With these in hand, you can obtain unbiased power spectra as follows, in the case of a spin-0 field for example:

.. code-block:: python
		
		> import pywiggle
		> import numpy as np

		> lmax = 4000
		> bin_edges = np.arange(40,lmax,40) # Choose bin edges

		# Initialize wiggle
		> w = Wiggle(lmax, bin_edges=bin_edges)
		# Register the SHT of a mask and identify it with a key
		> w.add_mask('mt1', mask_alm_t1)
		# Register another mask
		> w.add_mask('mt2', mask_alm_p2)
		# Register a beam to deconvolve from both fields
		> w.add_beam('b1', beam_fl)
		# Get the decoupled cross-Cls from the masked field SHTs
		> ret_TT = w.get_powers(alm_t1, alm_t2, 'mt1', 'mt2',
		                          return_theory_filter=False,
		     			  beam_id1='b1', beam_id2='b1')['TT']

This object can then be reused if the same masks are being re-used, which avoids re-calculation of mode-coupling matrices. The interface to ``get_powers`` is flexible enough to allow all auto- and cross- spectra of spin-0 and spin-2 fields.


Here ``dcls`` is the mode-decoupled unbiased power spectrum and ``th_filt`` is a matrix that can be dotted with a theory spectrum to obtain the binned theory to compare the power spectrum to (e.g. for inference):
    
		
.. code-block:: python
		
		> chisquare = get_chisquare(dcls,th_filt @ theory_cls,cinv)


Contributions
-------------

If you have write access to this repository, please:

1. create a new branch
2. push your changes to that branch
3. merge or rebase to get in sync with master
4. submit a pull request on github

If you do not have write access, create a fork of this repository and proceed as described above. 
