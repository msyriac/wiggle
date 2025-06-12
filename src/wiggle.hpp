/*
Copyright (c) 2025, Mathew S. Madhavacheril
// Licensed under the BSD 2-Clause License (see LICENSE file).

*/

#ifndef WIGGLE_H
#define WIGGLE_H

#include <stddef.h>
#include <cmath>
#include <vector>
#include <cstdint>   // for int64_t, uint64_t, etc.
#include <cstddef>   // for ssize_t


// Functions for wiggle in double precision
namespace wiggle {

  std::vector<double> compute_wigner_d_series(int lmax, int s1, int s2, double cos_theta);
  std::vector<double> compute_binned_wigner_d(int lmax, int s1, int s2, double cos_theta, int nbins,
					      const std::vector<int>& bin_indices,
					      const std::vector<double>& bin_weights);
 std::pair<std::vector<double>, std::vector<double>> compute_double_binned_wigner_d(
    int lmax,
    int s1,
    int s2,
    double cos_theta, int nbins,
    const std::vector<int>& bin_indices,
    const std::vector<double>& bin_weights,
    const std::vector<double>& bin_weights2);

 std::pair<std::vector<double>, std::vector<double>> compute_single_binned_wigner_d(
    int lmax,
    int s1,
    int s2,
    double cos_theta, int nbins,
    const std::vector<int>& bin_indices,
    const std::vector<double>& bin_weights);
  
  std::vector<double> compute_legendre_polynomials(int lmax, double x);

std::vector<double> bin_matrix_core(const double* data,
                                    const int64_t* x_bins,
                                    const int64_t* y_bins,
                                    const double* x_weights,
                                    const double* y_weights,
                                    int64_t nrows,
                                    int64_t ncols,
                                    int64_t nbins_x,
                                    int64_t nbins_y);


}
#endif
