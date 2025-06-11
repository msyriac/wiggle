/*

Copyright (c) 2025, Mathew S. Madhavacheril
// Licensed under the BSD 2-Clause License (see LICENSE file).

Some of the Wigner-d code below is a C++ port of code from Kendrick Smith.

*/

# include "wiggle.hpp"
# include "cassert"
#include <algorithm>
#include <vector>
#include <utility>          // std::pair
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <cstdint>   // for int64_t, uint64_t, etc.
#include <cstddef>   // for ssize_t

#ifdef _OPENMP
#include <omp.h>
#endif

// Anonymous namespace for non-public functions
namespace {



// Wigner-d recurrence coefficient
static inline double alpha(int l, int s1, int s2)
{
    if (l <= std::abs(s1) || l <= std::abs(s2))
        return 0.0;

    int l2 = l * l ;
    int ix = l2 - s1 * s1;
    int iy = l2 - s2 * s2;

    return std::sqrt((double)ix * (double)iy) / (double)l;
}

// Initializes Wigner-d at l = s2 (lowest possible l for given s1, s2)
static inline double wiginit(int s1, int s2, double cos_theta)
{
    double s12sign   = ((s1 + s2) % 2) ? -1.0 : 1.0;
    double prefactor = 1.0;

    if (std::abs(s1) > std::abs(s2)) {
        prefactor *= s12sign;
        std::swap(s1, s2);
    }

    if (s2 < 0) {
        prefactor *= s12sign;
        s1 = -s1;
        s2 = -s2;
    }

    int abs_s1 = std::abs(s1);
    assert(abs_s1 <= s2);

    for (int i = 1; i <= s2 - abs_s1; i++)
        prefactor *= std::sqrt((double)(s2 + abs_s1 + i) / (double)i);

    return prefactor *
           std::pow((1.0 + cos_theta) / 2.0, 0.5 * (double)(s2 + s1)) *
           std::pow((1.0 - cos_theta) / 2.0, 0.5 * (double)(s2 - s1));
}

// Recurrence step for Wigner-d: l -> l + 1
static inline void wigrec_step(int l, int s1, int s2, double cos_theta, double& wig_lo, double& wig_hi)
{
    double alpha_hi = alpha(l + 1, s1, s2);
    double alpha_lo = alpha(l, s1, s2);
    double beta     = (s1 == 0 || s2 == 0) ? 0.0 : ((double)s1 * (double)s2 / ((double)l * (l + 1)));

    double x = (2 * l + 1) * (cos_theta - beta) * wig_hi - alpha_lo * wig_lo;
    wig_lo = wig_hi;
    wig_hi = x / alpha_hi;
}

}


// Public functions exposed by this module
namespace wiggle
{

///
/// Compute Wigner-d functions for fixed s1, s2, single cos(theta),
/// over all l from 0 to lmax.
/// Used in wiggle for unbinned mode coupling computations.
std::vector<double> compute_wigner_d_series(int lmax, int s1, int s2, double cos_theta)
{
    std::vector<double> wigd(lmax + 1, 0.0);

    int l0 = std::max(std::abs(s1), std::abs(s2));
    if (l0 > lmax)
        return wigd;

    // Initialize at l = l0
    double wig_lo = 0.0;
    double wig_hi = wiginit(s1, s2, cos_theta);
    wigd[l0] = wig_hi;

    // Compute higher l
    for (int l = l0 + 1; l <= lmax; ++l) {
        wigrec_step(l - 1, s1, s2, cos_theta, wig_lo, wig_hi);
        wigd[l] = wig_hi;
    }

    return wigd;
}


///
/// Compute Wigner-d functions for fixed s1, s2, single cos(theta),
/// over all l from 0 to lmax but accumulated with weights into bins of specified
/// index. Typically not used in wiggle.
 std::vector<double> compute_binned_wigner_d(
    int lmax,
    int s1,
    int s2,
    double cos_theta, int nbins,
    const std::vector<int>& bin_indices,
    const std::vector<double>& bin_weights)
{
    std::vector<double> binned_wigd(nbins, 0.0);

    int l0 = std::max(std::abs(s1), std::abs(s2));
    if (l0 > lmax)
        return binned_wigd;

    double wig_lo = 0.0;
    double wig_hi = wiginit(s1, s2, cos_theta);

    // Accumulate initial l = l0
    if (l0 <= lmax) {
        int bin = bin_indices[l0];
        if (bin >= 0 && bin < nbins)
            binned_wigd[bin] += bin_weights[l0] * wig_hi;
    }

    for (int l = l0 + 1; l <= lmax; ++l) {
        wigrec_step(l - 1, s1, s2, cos_theta, wig_lo, wig_hi);
        int bin = bin_indices[l];
        if (bin >= 0 && bin < nbins)
            binned_wigd[bin] += bin_weights[l] * wig_hi;
    }

    return binned_wigd;
}

///
/// Compute Wigner-d functions for fixed s1, s2, single cos(theta),
/// over all l from 0 to lmax but accumulated with weights into two sets of bins of specified
/// indices. Used in wiggle for binned mode coupling computations.
 std::pair<std::vector<double>, std::vector<double>> compute_double_binned_wigner_d(
    int lmax,
    int s1,
    int s2,
    double cos_theta, int nbins,
    const std::vector<int>& bin_indices,
    const std::vector<double>& bin_weights,
    const std::vector<double>& bin_weights2)
{

  std::vector<double> binned_wigd(nbins, 0.0);
    std::vector<double> binned_wigd2(nbins, 0.0);

    int l0 = std::max(std::abs(s1), std::abs(s2));
    if (l0 > lmax)
        return {binned_wigd, binned_wigd2};

    double wig_lo = 0.0;
    double wig_hi = wiginit(s1, s2, cos_theta);

    
    // Accumulate initial l = l0
    if (l0 <= lmax) {
        int bin = bin_indices[l0];
        if (bin >= 0 && bin < nbins)
	  {
            binned_wigd[bin] += bin_weights[l0] * wig_hi;
            binned_wigd2[bin] += bin_weights2[l0] * wig_hi;
	  }
    }

    for (int l = l0 + 1; l <= lmax; ++l) {
        wigrec_step(l - 1, s1, s2, cos_theta, wig_lo, wig_hi);
        int bin = bin_indices[l];
        if (bin >= 0 && bin < nbins)
	  {
            binned_wigd[bin] += bin_weights[l] * wig_hi;
            binned_wigd2[bin] += bin_weights2[l] * wig_hi;
	  }
    }

    return {binned_wigd, binned_wigd2};
}

///
/// Compute Wigner-d functions for fixed s1, s2, single cos(theta),
/// over all l from 0 to lmax but accumulated with weights into bins of specified
/// index, but also return the unbinned version along with it.
/// Typically not used in wiggle.
 std::pair<std::vector<double>, std::vector<double>> compute_single_binned_wigner_d(
    int lmax,
    int s1,
    int s2,
    double cos_theta, int nbins,
    const std::vector<int>& bin_indices,
    const std::vector<double>& bin_weights)
{

  std::vector<double> binned_wigd(nbins, 0.0);
    std::vector<double> wigd(lmax + 1, 0.0);

    int l0 = std::max(std::abs(s1), std::abs(s2));
    if (l0 > lmax)
        return {binned_wigd, wigd};

    double wig_lo = 0.0;
    double wig_hi = wiginit(s1, s2, cos_theta);
    wigd[l0] = wig_hi;

    
    // Accumulate initial l = l0
    if (l0 <= lmax) {
        int bin = bin_indices[l0];
        if (bin >= 0 && bin < nbins)
	  {
            binned_wigd[bin] += bin_weights[l0] * wig_hi;
	  }
    }

    for (int l = l0 + 1; l <= lmax; ++l) {
        wigrec_step(l - 1, s1, s2, cos_theta, wig_lo, wig_hi);
        wigd[l] = wig_hi;
        int bin = bin_indices[l];
        if (bin >= 0 && bin < nbins)
	  {
            binned_wigd[bin] += bin_weights[l] * wig_hi;
	  }
    }

    return {binned_wigd, wigd};
}

///
/// Compute Legendre polynomials for single cos(theta)
/// Bit faster than Wigner-d 00 recursion
  std::vector<double> compute_legendre_polynomials(int lmax, double x)
{
    std::vector<double> P(lmax + 1);
    P[0] = 1.0;
    if (lmax >= 1) P[1] = x;

    for (int l = 1; l < lmax; ++l)
        P[l + 1] = ((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1);

    return P;
}

///
/// Bin a matrix along both dimensions
/// 

std::vector<double> bin_matrix_core(const double*      data,
                                   const int64_t*     y_bins,
                                   const int64_t*     x_bins,
                                   const double*      w_y,
                                   const double*      w_x,
                                   int64_t            Ny,
                                   int64_t            Nx,
                                   int64_t            nbins_y,
                                   int64_t            nbins_x)
{
    if (Ny <= 0 || Nx <= 0)
        throw std::invalid_argument("Matrix dimensions must be positive");
    if (nbins_y <= 0 || nbins_x <= 0)
        throw std::invalid_argument("Number of bins must be positive");

    std::vector<double> out(static_cast<size_t>(nbins_y * nbins_x), 0.0);

#pragma omp parallel for
    for (int64_t i = 0; i < Ny; ++i) {
        int64_t by = y_bins[i];
        if (by < 0 || by >= nbins_y) continue;  // invalid row bin

        double wy = w_y[i];
        const double* row = data + i * Nx;

        for (int64_t j = 0; j < Nx; ++j) {
            int64_t bx = x_bins[j];
            if (bx < 0 || bx >= nbins_x) continue;  // invalid col bin

            double wx = w_x[j];
            double val = row[j] * wy * wx;
            size_t idx = static_cast<size_t>(by * nbins_x + bx); // row‑major
#pragma omp atomic
            out[idx] += val;
        }
    }

    return out; // move elided (NRVO)
}
  
}

