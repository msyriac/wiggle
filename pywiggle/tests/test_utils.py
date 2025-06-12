import numpy as np
import pytest
from pywiggle import utils, _wiggle

def test_normalize_weights_per_bin():
    nbins = 4
    bin_indices = np.array([0, 0, 1, 1, 1, 2, -1, 3, 3, 3])
    weights =     np.array([2., 3., 1., 4., 5., 10., 7., 6., 3., 1.])

    # Expected behavior:
    # Bin 0: weights = [2, 3] → sum = 5 → normalized = [0.4, 0.6]
    # Bin 1: weights = [1, 4, 5] → sum = 10 → normalized = [0.1, 0.4, 0.5]
    # Bin 2: weights = [10] → sum = 10 → normalized = [1.0]
    # Bin 3: weights = [6, 3, 1] → sum = 10 → normalized = [0.6, 0.3, 0.1]
    # Bin -1: ignored → normalized = 0

    expected = np.array([
        2/5, 3/5,        # bin 0
        1/10, 4/10, 5/10, # bin 1
        1.0,              # bin 2
        0.0,              # bin -1 (ignored)
        6/10, 3/10, 1/10  # bin 3
    ])

    result = utils.normalize_weights_per_bin(nbins, bin_indices, weights)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("Test passed.")


def test_bin_multipole_array_nontrivial():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bin_indices = np.array([0, 1, 1, -1, 0, 2])  # Note: -1 should be ignored

    # Explanation:
    # bin 0: arr[0] + arr[4] = 1.0 + 5.0 = 6.0
    # bin 1: arr[1] + arr[2] = 2.0 + 3.0 = 5.0
    # bin 2: arr[5] = 6.0
    # bin -1: arr[3] = 4.0 → ignored

    expected = np.array([6.0, 5.0, 6.0])

    result2 = utils.bin_array(arr, bin_indices, nbins=3)
    expected2 = np.array([6.0, 5.0, 6.0])  # padded with zeros
    assert np.allclose(result2, expected2), f"Expected {expected2}, got {result2}"
    
    # Test with explicit nbins (larger than needed)
    result2 = utils.bin_array(arr, bin_indices, nbins=5)
    expected2 = np.array([6.0, 5.0, 6.0, 0.0, 0.0])  # padded with zeros
    assert np.allclose(result2, expected2), f"Expected {expected2}, got {result2}"




def test_uniform_weights_and_bins():
    mat = np.array([[1, 2],
                    [3, 4]])
    y_bins = np.array([0, 0])   # both rows go to bin 0
    x_bins = np.array([0, 1])   # first column to bin 0, second to bin 1
    w_y = np.ones(2)
    w_x = np.ones(2)

    result = _wiggle.bin_matrix(mat, y_bins, x_bins, w_y, w_x, nbins_y=1, nbins_x=2)
    expected = np.array([[4, 6]])
    np.testing.assert_array_equal(result, expected)


def test_weighted_sum_with_different_bins():
    mat = np.array([[1, 2],
                    [3, 4],
                    [5, 6]])
    y_bins = np.array([0, 1, 1])
    x_bins = np.array([0, 1])
    w_y = np.array([1, 10, 100])
    w_x = np.array([1, 0.1])

    result = _wiggle.bin_matrix(mat, y_bins, x_bins, w_y, w_x, nbins_y=2, nbins_x=2)

    # Row 0 (bin 0): [1*1*1, 2*1*0.1] = [1, 0.2]
    # Row 1 (bin 1): [3*10*1, 4*10*0.1] = [30, 4]
    # Row 2 (bin 1): [5*100*1, 6*100*0.1] = [500, 60]
    expected = np.array([[1.0, 0.2],
                         [530.0, 64.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_out_of_bounds_bins_are_ignored():
    mat = np.array([[1, 2],
                    [3, 4]])
    y_bins = np.array([0, -1])  # second row invalid
    x_bins = np.array([1, 3])   # column 3 is out of bounds
    w_y = np.ones(2)
    w_x = np.ones(2)

    result = _wiggle.bin_matrix(mat, y_bins, x_bins, w_y, w_x, nbins_y=1, nbins_x=3)

    # Only mat[0,0] counts (y=0, x=1): value = 1
    expected = np.zeros((1, 3))
    expected[0, 1] = 1
    np.testing.assert_array_equal(result, expected)


def test_empty_matrix_raises():
    mat = np.zeros((0, 0))
    y_bins = np.array([], dtype=np.int64)
    x_bins = np.array([], dtype=np.int64)
    w_y = np.array([], dtype=np.float64)
    w_x = np.array([], dtype=np.float64)

    with pytest.raises(ValueError):
        _wiggle.bin_matrix(mat, y_bins, x_bins, w_y, w_x, nbins_y=1, nbins_x=1)


def test_large_random_case():
    np.random.seed(0)
    Ny, Nx = 100, 80
    mat = np.random.rand(Ny, Nx)
    y_bins = np.random.randint(0, 10, size=Ny)
    x_bins = np.random.randint(0, 5, size=Nx)
    w_y = np.random.rand(Ny)
    w_x = np.random.rand(Nx)

    result = _wiggle.bin_matrix(mat, y_bins, x_bins, w_y, w_x, nbins_y=10, nbins_x=5)

    assert result.shape == (10, 5)
    expected_total = np.sum(mat * w_y[:, None] * w_x[None, :])
    np.testing.assert_allclose(result.sum(), expected_total, rtol=1e-10)
