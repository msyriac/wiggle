import numpy as np
import pywiggle
from pixell import bench
from pywiggle import utils

lmax = 2048


def main():
    """Benchmark wiggle vs ducc0 multi-mask TT coupling matrices at lmax=2048."""
    mcls = np.ones((10, 2 * lmax + 1))

    with bench.show("mcm"):
        w = pywiggle.Wiggle(lmax)
        w.get_coupling_matrix_from_mask_cls(mcls, "TT")

    with bench.show("ducc"):
        utils._mcm00_ducc(mcls, lmax, nthreads=12)


if __name__ == "__main__":
    main()
