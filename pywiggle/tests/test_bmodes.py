import numpy as np
from pixell import enmap, enplot, curvedsky as cs, utils
import matplotlib.pyplot as plt

import camb
from camb import model

def get_camb_spectra(lmax=512, tensor=True, ns=0.965, As=2e-9, r=0.1):
    """
    Returns CMB Cls [TT, EE, BB, TE] from CAMB with low accuracy for fast testing.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = tensor
    pars.AccuracyBoost = 0.1
    pars.lAccuracyBoost = 0.1
    pars.HighAccuracyDefault = False

    results = camb.get_results(pars)
    cls = results.get_total_cls(lmax=lmax)

    # cls has shape (lmax+1, 4): TT, EE, BB, TE
    ps = np.zeros((3, 3, lmax+1))
    ps[0, 0] = cls[:, 0]  # TT
    ps[1, 1] = cls[:, 1]  # EE
    ps[2, 2] = cls[:, 2]  # BB
    ps[0, 1] = ps[1, 0] = cls[:, 3]  # TE

    return ps


def test_recover_tensor_Bmode():
    # --- 1. Sim config ---
    res = 4.0 / 60. # deg
    shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res))
    lmax = 1000
    area_deg2 = 100
    radius_deg = np.sqrt(area_deg2 / np.pi)
    radius_rad = np.deg2rad(radius_deg)

    # --- 2. Generate pure tensor (B-mode) Cls ---
    ps = get_camb_spectra(lmax=lmax, r=0.05)  # r sets BB amplitude

    # --- 3. Simulate polarization map ---
    alm = cs.rand_alm(ps, lmax=lmax)
    polmap = cs.alm2map(alm[1:], shape, wcs, spin=2)  # only Q,U
    Q, U = polmap

    # --- 4. Make apodized circular mask of 100 deg² ---
    y, x = enmap.posmap(shape, wcs)
    r = np.sqrt(x**2 + y**2)
    mask = np.clip((radius_rad - r) / (0.1 * radius_rad), 0, 1)
    mask = utils.apod(mask, 10, 'C1')  # smooth edge

    # --- 5. Run purification ---
    from your_module import pure_EB  # Adjust to actual location
    pureE, pureB = pure_EB(Q, U, mask, is_healpix=False)

    # --- 6. Compute power spectrum and compare ---
    bpow = cs.alm2cl(pureB)
    input_bb = ps[2, 2]

    plt.figure()
    ell = np.arange(len(bpow))
    plt.plot(ell, input_bb[:len(ell)], label='Input BB')
    plt.plot(ell, bpow, label='Recovered pure B')
    plt.xlim(10, 300)
    plt.ylim(1e-8, 1e-3)
    plt.yscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell^{BB}$')
    plt.legend()
    plt.title('Pure B-mode recovery test (100 deg$^2$ mask)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 7. Assert success in rough sense ---
    band = (30 < ell) & (ell < 150)
    ratio = np.mean(bpow[band] / input_bb[band])
    assert 0.5 < ratio < 2.0, f"Recovery failed: ratio = {ratio:.2f}"

# Run test
test_recover_tensor_Bmode()
