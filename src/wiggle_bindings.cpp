/*

Copyright (c) 2025, Mathew S. Madhavacheril
// Licensed under the BSD 2-Clause License (see LICENSE file).

*/
// wiggle_bindings.cpp
// pybind11 glue for wiggle.

#ifdef _OPENMP
#include <omp.h>
#endif

#include <pybind11/pybind11.h>
#include "wiggle.hpp"
#include <pybind11/numpy.h>
#include <memory>
#include <utility>
#include <vector>
// #include <chrono> // optional timing debugging
#include <iostream>
#include <algorithm>
#include <cstdint>   // for int64_t, uint64_t, etc.
#include <cstddef>   // for ssize_t


namespace py = pybind11;
namespace fg = wiggle;

// Python binding for bin_matrix_core passing and returning numpy arrays
py::array bin_matrix_py(py::array_t<double,  py::array::c_style | py::array::forcecast> mat,
                        py::array_t<int64_t, py::array::c_style | py::array::forcecast> y_bins,
                        py::array_t<int64_t, py::array::c_style | py::array::forcecast> x_bins,
                        py::array_t<double,  py::array::c_style | py::array::forcecast> w_y,
                        py::array_t<double,  py::array::c_style | py::array::forcecast> w_x,
                        int64_t nbins_y,
                        int64_t nbins_x)
{
    // --- basic validation -----------------------------------------------------
    if (mat.ndim() != 2)
        throw std::invalid_argument("`mat` must be 2‑D");
    const int64_t Ny = mat.shape(0);
    const int64_t Nx = mat.shape(1);
    if (y_bins.ndim() != 1 || y_bins.shape(0) != Ny)
        throw std::invalid_argument("Length of y_bins must equal number of rows in mat");
    if (x_bins.ndim() != 1 || x_bins.shape(0) != Nx)
        throw std::invalid_argument("Length of x_bins must equal number of cols in mat");
    if (w_y.ndim() != 1 || w_y.shape(0) != Ny)
        throw std::invalid_argument("Length of w_y must equal number of rows in mat");
    if (w_x.ndim() != 1 || w_x.shape(0) != Nx)
        throw std::invalid_argument("Length of w_x must equal number of cols in mat");

    // --- call core ------------------------------------------------------------
    std::vector<double> result = fg::bin_matrix_core(mat.data(),
                                                 y_bins.data(),
                                                 x_bins.data(),
                                                 w_y.data(),
                                                 w_x.data(),
                                                 Ny,
                                                 Nx,
                                                 nbins_y,
                                                 nbins_x);

    // --- wrap vector into NumPy array without copy ---------------------------
    auto* vec_ptr = new std::vector<double>(std::move(result));
    double* data_ptr = vec_ptr->data();

    // Capsule to free memory when Python GC releases the array
    py::capsule owner(vec_ptr, [](void* p){ delete reinterpret_cast<std::vector<double>*>(p); });

    std::vector<ssize_t> shape   = {nbins_y, nbins_x};
    std::vector<ssize_t> strides = {
      static_cast<ssize_t>(sizeof(double) * nbins_x),
      static_cast<ssize_t>(sizeof(double))
    };
    return py::array(shape, strides, data_ptr, owner);
}

// More Python bindings

PYBIND11_MODULE(_wiggle, m) {
    m.doc() = "Python bindings for wiggle";

m.def(
    "_compute_wigner_d_matrix",
    [](int lmax, int s1, int s2, py::array_t<double, py::array::c_style | py::array::forcecast> cos_theta_np) {
        auto buf = cos_theta_np.request();

	if (buf.ndim != 1)
            throw std::runtime_error("cos_theta must be a 1D array");

        const double* cos_theta = static_cast<const double*>(buf.ptr);
        size_t ntheta = static_cast<size_t>(buf.shape[0]);
        size_t ncols = lmax + 1;

        // Allocate flat output array (owned by NumPy from the start)
        py::array_t<double> result({ntheta, ncols});
        auto r = result.mutable_unchecked<2>();
	
	// std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        // Fill the result array in-place using compute_wigner_d_series
        #pragma omp parallel for
        for (ssize_t i = 0; i < static_cast<ssize_t>(ntheta); ++i) {
	  auto row = fg::compute_wigner_d_series(lmax, s1, s2, cos_theta[i]);
            for (size_t l = 0; l < row.size(); ++l)
                r(i, l) = row[l];

        }

	// std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	// std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	

        return result;
    },
    py::arg("lmax"),
    py::arg("s1"),
    py::arg("s2"),
    py::arg("cos_theta"));



m.def(
    "_compute_binned_wigner_d",
    [](int lmax, int s1, int s2,
       py::array_t<double, py::array::c_style | py::array::forcecast> cos_theta_np,
       int nbins,
       py::array_t<int, py::array::c_style | py::array::forcecast> bin_indices_np,
       py::array_t<double, py::array::c_style | py::array::forcecast> weights_np) {
        
        // Extract input buffers
        auto cos_buf = cos_theta_np.request();
        auto bins_buf = bin_indices_np.request();
        auto weights_buf = weights_np.request();

        const double* cos_theta = static_cast<const double*>(cos_buf.ptr);
        const int* bin_indices = static_cast<const int*>(bins_buf.ptr);
        const double* weights = static_cast<const double*>(weights_buf.ptr);

        size_t ntheta = static_cast<size_t>(cos_buf.shape[0]);
        size_t numbins = static_cast<size_t>(nbins);

        if (weights_buf.shape[0] < static_cast<size_t>(lmax + 1)) {
            throw std::invalid_argument("weights array must have length at least lmax + 1");
        }

        // Allocate output array (ntheta, nbins)
        py::array_t<double> result({ntheta, numbins});
        auto r = result.mutable_unchecked<2>();

        // Compute each row in parallel
        #pragma omp parallel for
        for (ssize_t i = 0; i < static_cast<ssize_t>(ntheta); ++i) {
	  std::vector<double> row = fg::compute_binned_wigner_d(
								lmax, s1, s2, cos_theta[i], nbins,
								std::vector<int>(bin_indices, bin_indices + bins_buf.shape[0]),
								std::vector<double>(weights, weights + weights_buf.shape[0])
            );

            for (size_t j = 0; j < nbins; ++j)
                r(i, j) = row[j];
        }

        return result;
    },
    py::arg("lmax"),
    py::arg("s1"),
    py::arg("s2"),
    py::arg("cos_theta"),
    py::arg("nbins"),
    py::arg("bin_indices"),
    py::arg("weights"));
 

 m.def(
    "_compute_double_binned_wigner_d",
    [](int lmax, int s1, int s2,
       py::array_t<double, py::array::c_style | py::array::forcecast> cos_theta_np,
       int nbins,
       py::array_t<int, py::array::c_style | py::array::forcecast> bin_indices_np,
       py::array_t<double, py::array::c_style | py::array::forcecast> weights1_np,
       py::array_t<double, py::array::c_style | py::array::forcecast> weights2_np) {

        // Extract buffers
        auto cos_buf = cos_theta_np.request();
        auto bins_buf = bin_indices_np.request();
        auto w1_buf = weights1_np.request();
        auto w2_buf = weights2_np.request();

        const double* cos_theta = static_cast<const double*>(cos_buf.ptr);
        const int* bin_indices = static_cast<const int*>(bins_buf.ptr);
        const double* w1 = static_cast<const double*>(w1_buf.ptr);
        const double* w2 = static_cast<const double*>(w2_buf.ptr);

        size_t ntheta = static_cast<size_t>(cos_buf.shape[0]);

        if (w1_buf.shape[0] < static_cast<size_t>(lmax + 1) || 
            w2_buf.shape[0] < static_cast<size_t>(lmax + 1)) {
            throw std::invalid_argument("weights arrays must have length at least lmax + 1");
        }

        // Output arrays: (ntheta, nbins)
        py::array_t<double> result1({ntheta, static_cast<size_t>(nbins)});
        py::array_t<double> result2({ntheta, static_cast<size_t>(nbins)});

        auto r1 = result1.mutable_unchecked<2>();
        auto r2 = result2.mutable_unchecked<2>();

        #pragma omp parallel for
        for (ssize_t i = 0; i < static_cast<ssize_t>(ntheta); ++i) {
            auto [b1, b2] = fg::compute_double_binned_wigner_d(
                lmax, s1, s2, cos_theta[i], nbins,
                std::vector<int>(bin_indices, bin_indices + bins_buf.shape[0]),
                std::vector<double>(w1, w1 + w1_buf.shape[0]),
                std::vector<double>(w2, w2 + w2_buf.shape[0])
            );

            for (size_t j = 0; j < static_cast<size_t>(nbins); ++j) {
                r1(i, j) = b1[j];
                r2(i, j) = b2[j];
            }
        }

        // Return as tuple
        return py::make_tuple(result1, result2);
    },
    py::arg("lmax"),
    py::arg("s1"),
    py::arg("s2"),
    py::arg("cos_theta"),
    py::arg("nbins"),
    py::arg("bin_indices"),
    py::arg("weights1"),
    py::arg("weights2"));

 m.def(
    "_compute_single_binned_wigner_d",
    [](int lmax, int s1, int s2,
       py::array_t<double, py::array::c_style | py::array::forcecast> cos_theta_np,
       int nbins,
       py::array_t<int, py::array::c_style | py::array::forcecast> bin_indices_np,
       py::array_t<double, py::array::c_style | py::array::forcecast> weights1_np) {

        // Extract buffers
        auto cos_buf = cos_theta_np.request();
        auto bins_buf = bin_indices_np.request();
        auto w1_buf = weights1_np.request();

        const double* cos_theta = static_cast<const double*>(cos_buf.ptr);
        const int* bin_indices = static_cast<const int*>(bins_buf.ptr);
        const double* w1 = static_cast<const double*>(w1_buf.ptr);

        size_t ntheta = static_cast<size_t>(cos_buf.shape[0]);
        size_t ncols = lmax + 1;

        if (w1_buf.shape[0] < static_cast<size_t>(lmax + 1) ) {
            throw std::invalid_argument("weights arrays must have length at least lmax + 1");
        }

        // Output arrays: (ntheta, nbins)
        py::array_t<double> result1({ntheta, static_cast<size_t>(nbins)});

        auto r1 = result1.mutable_unchecked<2>();

        py::array_t<double> result({ntheta, ncols});
        auto r = result.mutable_unchecked<2>();
	
        #pragma omp parallel for
        for (ssize_t i = 0; i < static_cast<ssize_t>(ntheta); ++i) {
            auto [b1, row] = fg::compute_single_binned_wigner_d(
                lmax, s1, s2, cos_theta[i], nbins,
                std::vector<int>(bin_indices, bin_indices + bins_buf.shape[0]),
                std::vector<double>(w1, w1 + w1_buf.shape[0])
            );

            for (size_t j = 0; j < static_cast<size_t>(nbins); ++j) {
                r1(i, j) = b1[j];
            }

            for (size_t l = 0; l < row.size(); ++l)
                r(i, l) = row[l];
	    
        }

        // Return as tuple
        return py::make_tuple(result1, result);
    },
    py::arg("lmax"),
    py::arg("s1"),
    py::arg("s2"),
    py::arg("cos_theta"),
    py::arg("nbins"),
    py::arg("bin_indices"),
    py::arg("weights1"));

  m.def(
        "_compute_legendre_matrix",
        [](int lmax, py::array_t<double, py::array::c_style | py::array::forcecast> cos_theta_np) {
            auto buf = cos_theta_np.request();

            if (buf.ndim != 1)
                throw std::runtime_error("cos_theta must be a 1D array");

            const double* cos_theta = static_cast<const double*>(buf.ptr);
            size_t ntheta = static_cast<size_t>(buf.shape[0]);
            size_t ncols = lmax + 1;

            py::array_t<double> result({ntheta, ncols});
            auto r = result.mutable_unchecked<2>();

            #pragma omp parallel for
            for (ssize_t i = 0; i < static_cast<ssize_t>(ntheta); ++i) {
	      std::vector<double> row = fg::compute_legendre_polynomials(lmax, cos_theta[i]);
                for (size_t l = 0; l < row.size(); ++l)
                    r(i, l) = row[l];
            }

            return result;
        },
        py::arg("lmax"),
        py::arg("cos_theta"));

  m.def("bin_matrix", &bin_matrix_py,
          py::arg("mat"), py::arg("y_bins"), py::arg("x_bins"),
          py::arg("w_y"), py::arg("w_x"),
      py::arg("nbins_y"), py::arg("nbins_x"));

}

