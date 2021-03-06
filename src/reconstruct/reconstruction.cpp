//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
//! \file reconstruction.cpp
//  \brief

#include "reconstruct/reconstruction.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

Reconstruction::Reconstruction(std::weak_ptr<MeshBlock> wpmb, ParameterInput *pin)
    : characteristic_projection{false}, uniform{true, true, true, true},
      // read fourth-order solver switches
      correct_ic{pin->GetOrAddBoolean("parthenon/time", "correct_ic", false)},
      correct_err{pin->GetOrAddBoolean("parthenon/time", "correct_err", false)},
      pmy_block_{wpmb} {
  // Read and set type of spatial reconstruction
  // --------------------------------
  std::string input_recon = pin->GetOrAddString("parthenon/mesh", "xorder", "2");
  // meshblock
  std::shared_ptr<MeshBlock> pmb = wpmb.lock();
  // Avoid pmb indirection
  const IndexDomain entire = IndexDomain::entire;
  const IndexDomain interior = IndexDomain::interior;
  const IndexShape &cellbounds = pmb->cellbounds;

  if (input_recon == "1") {
    xorder = 1;
  } else if (input_recon == "2") {
    xorder = 2;
  } else if (input_recon == "2c") {
    xorder = 2;
    characteristic_projection = true;
  } else if (input_recon == "3") {
    // PPM approximates interfaces with 4th-order accurate stencils, but use xorder=3
    // to denote that the overall scheme is "between 2nd and 4th" order w/o flux terms
    xorder = 3;
  } else if (input_recon == "3c") {
    xorder = 3;
    characteristic_projection = true;
  } else if ((input_recon == "4") || (input_recon == "4c")) {
    xorder = 4;
    if (input_recon == "4c") characteristic_projection = true;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
        << "xorder=" << input_recon << " not valid choice for reconstruction"
        << std::endl;
    PARTHENON_FAIL(msg);
  }
  // Check for incompatible choices with broader solver configuration
  // --------------------------------

  // check for necessary number of ghost zones for PPM w/o fourth-order flux corrections
  if (xorder == 3) {
    int req_nghost = 3;
    if (Globals::nghost < req_nghost) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
          << "xorder=" << input_recon
          << " (PPM) reconstruction selected, but nghost=" << Globals::nghost << std::endl
          << "Rerun with --nghost=XXX with XXX > " << req_nghost - 1 << std::endl;
      PARTHENON_FAIL(msg);
    }
  }

  // perform checks of fourth-order solver configuration restrictions:
  if (xorder == 4) {
    // Uniform, Cartesian mesh with square cells (dx1f=dx2f=dx3f)
    if (pmb->block_size.x1rat != 1.0 || pmb->block_size.x2rat != 1.0 ||
        pmb->block_size.x3rat != 1.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
          << "Selected time/xorder=" << input_recon << " flux calculations"
          << " require a uniform (x1rat=x2rat=x3rat=1.0), " << std::endl
          << "Carteisan mesh with square cells. Rerun with uniform cell spacing "
          << std::endl
          << "Current values are:" << std::endl
          << std::scientific
          << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1)
          << "x1rat= " << pmb->block_size.x1rat << std::endl
          << "x2rat= " << pmb->block_size.x2rat << std::endl
          << "x3rat= " << pmb->block_size.x3rat << std::endl;
      PARTHENON_FAIL(msg);
    }
    Real dx_i = pmb->coords.dx1f(cellbounds.is(interior));
    Real dx_j = pmb->coords.dx2f(cellbounds.js(interior));
    Real dx_k = pmb->coords.dx3f(cellbounds.ks(interior));
    // Note, probably want to make the following condition less strict (signal warning
    // for small differences due to floating-point issues) but upgrade to error for
    // large deviations from a square mesh. Currently signals a warning for each
    // MeshBlock with non-square cells.
    if ((pmb->block_size.nx2 > 1 && dx_i != dx_j) ||
        (pmb->block_size.nx3 > 1 && dx_j != dx_k)) {
      // It is possible for small floating-point differences to arise despite equal
      // analytic values for grid spacings in the coordinates.cpp calculation of:
      // Real dx=(block_size.x1max-block_size.x1min)/(ie-is+1);
      // due to the 3x rounding operations in numerator, e.g.
      // float(float(x1max) - float((x1min))
      // if mesh/x1max != mesh/x2max, etc. and/or if an asymmetric MeshBlock
      // decomposition is used
      if (Globals::my_rank == 0) {
        // std::stringstream msg;
        std::cout << "### Warning in Reconstruction constructor" << std::endl
                  << "Selected time/xorder=" << input_recon << " flux calculations"
                  << " require a uniform, Carteisan mesh with" << std::endl
                  << "square cells (dx1f=dx2f=dx3f). "
                  << "Change mesh limits and/or number of cells for equal spacings\n"
                  << "Current values are:" << std::endl
                  << std::scientific
                  << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1)
                  << "dx1f=" << dx_i << std::endl
                  << "dx2f=" << dx_j << std::endl
                  << "dx3f=" << dx_k << std::endl;
        // PARTHENON_FAIL(msg);
      }
    }
    if (pmb->pmy_mesh->multilevel) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
          << "Selected time/xorder=" << input_recon << " flux calculations"
          << " currently does not support SMR/AMR " << std::endl;
      PARTHENON_FAIL(msg);
    }

    // check for necessary number of ghost zones for PPM w/ fourth-order flux corrections
    int req_nghost = 4;
    // conversion is added, Globals::nghost>=6
    if (Globals::nghost < req_nghost) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
          << "time/xorder=" << input_recon
          << " reconstruction selected, but nghost=" << Globals::nghost << std::endl
          << "Rerun with --nghost=XXX with XXX > " << req_nghost - 1 << std::endl;
      PARTHENON_FAIL(msg);
    }
  }

  // for all coordinate systems, nonuniform geometric spacing or user-defined
  // MeshGenerator ---> use nonuniform reconstruction weights and limiter terms
  if (pmb->block_size.x1rat != 1.0) uniform[X1DIR] = false;
  if (pmb->block_size.x2rat != 1.0) uniform[X2DIR] = false;
  if (pmb->block_size.x3rat != 1.0) uniform[X3DIR] = false;

  // Uniform mesh with --coord=cartesian or GR: Minkowski, Schwarzschild, Kerr-Schild,
  // GR-User will use the uniform Cartesian limiter and reconstruction weights
  // TODO(c-white): use modified version of curvilinear PPM reconstruction weights and
  // limiter formulations for Schwarzschild, Kerr metrics instead of Cartesian-like wghts

  // Allocate memory for scratch arrays used in PLM and PPM
  int nc1 = cellbounds.ncellsi(entire);
  scr01_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
  scr02_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);

  scr1_ni_ = ParArrayND<Real>(PARARRAY_TEMP, NWAVE, nc1);
  scr2_ni_ = ParArrayND<Real>(PARARRAY_TEMP, NWAVE, nc1);
  scr3_ni_ = ParArrayND<Real>(PARARRAY_TEMP, NWAVE, nc1);
  scr4_ni_ = ParArrayND<Real>(PARARRAY_TEMP, NWAVE, nc1);

  if ((xorder == 3) || (xorder == 4)) {
    auto &coords = pmb->coords;
    scr03_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr04_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr05_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr06_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr07_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr08_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr09_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr10_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr11_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr12_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr13_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    scr14_i_ = ParArrayND<Real>(PARARRAY_TEMP, nc1);

    scr5_ni_ = ParArrayND<Real>(PARARRAY_TEMP, NWAVE, nc1);
    scr6_ni_ = ParArrayND<Real>(PARARRAY_TEMP, NWAVE, nc1);
    scr7_ni_ = ParArrayND<Real>(PARARRAY_TEMP, NWAVE, nc1);
    scr8_ni_ = ParArrayND<Real>(PARARRAY_TEMP, NWAVE, nc1);

    // Precompute PPM coefficients in x1-direction ---------------------------------------
    c1i = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    c2i = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    c3i = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    c4i = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    c5i = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    c6i = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    hplus_ratio_i = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    hminus_ratio_i = ParArrayND<Real>(PARARRAY_TEMP, nc1);

    // Greedily allocate tiny 4x4 matrix + 4x1 vectors (RHS, solution, and permutation
    // indices) in case PPMx1 and/or PPMx2 require them for computing the curvilinear
    // coorddinate reconstruction weights. Same data structures are reused at each spatial
    // index (i or j) and for both PPMx1 and PPMx2 weight calculations:
    constexpr int kNrows = 4; // = [i-i_L, i+i_R] stencil of reconstruction
    constexpr int kNcols = 4; // = [0, p-1], p=order of reconstruction
    // system in Mignone equation 21
    Real **beta = new Real *[kNrows];
    for (int i = 0; i < kNrows; ++i) {
      beta[i] = new Real[kNcols];
    }

    // zero-curvature PPM limiter does not depend on mesh uniformity:
    const IndexDomain interior = IndexDomain::interior;
    for (int i = (pmb->cellbounds.is(interior)) - 1;
         i <= (pmb->cellbounds.ie(interior)) + 1; ++i) {
      // h_plus = 3.0;
      // h_minus = 3.0;
      // Ratios are = 2 for Cartesian coords, as in the original PPM limiter's
      // overshoot conditions
      hplus_ratio_i(i) = 2.0;
      hminus_ratio_i(i) = 2.0;
    }
    // 4th order reconstruction weights along Cartesian-like x1 w/ uniform spacing
    if (uniform[X1DIR]) {
#pragma omp simd
      for (int i = cellbounds.is(entire); i <= cellbounds.ie(entire); ++i) {
        // reducing general formula in ppm.cpp corresonds to Mignone eq B.4 weights:
        // (-1/12, 7/12, 7/12, -1/12)
        c1i(i) = 0.5;
        c2i(i) = 0.5;
        c3i(i) = 0.5;
        c4i(i) = 0.5;
        c5i(i) = 1.0 / 6.0;
        c6i(i) = -1.0 / 6.0;
      }
    } else { // coeffcients along Cartesian-like x1 with nonuniform mesh spacing
#pragma omp simd
      for (int i = pmb->cellbounds.is(entire) + 1; i <= pmb->cellbounds.ie(entire) - 1;
           ++i) {
        Real dx_im1 = coords.dx1f(i - 1);
        Real dx_i = coords.dx1f(i);
        Real dx_ip1 = coords.dx1f(i + 1);
        Real qe = dx_i / (dx_im1 + dx_i + dx_ip1); // Outermost coeff in CW eq 1.7
        c1i(i) = qe * (2.0 * dx_im1 + dx_i) / (dx_ip1 + dx_i); // First term in CW eq 1.7
        c2i(i) = qe * (2.0 * dx_ip1 + dx_i) / (dx_im1 + dx_i); // Second term in CW eq 1.7
        if (i >
            pmb->cellbounds.is(entire) + 1) { // c3-c6 are not computed in first iteration
          Real dx_im2 = coords.dx1f(i - 2);
          Real qa = dx_im2 + dx_im1 + dx_i + dx_ip1;
          Real qb = dx_im1 / (dx_im1 + dx_i);
          Real qc = (dx_im2 + dx_im1) / (2.0 * dx_im1 + dx_i);
          Real qd = (dx_ip1 + dx_i) / (2.0 * dx_i + dx_im1);
          qb = qb + 2.0 * dx_i * qb / qa * (qc - qd);
          c3i(i) = 1.0 - qb;
          c4i(i) = qb;
          c5i(i) = dx_i / qa * qd;
          c6i(i) = -dx_im1 / qa * qc;
        }
      }
    }

    // Precompute PPM coefficients in x2-direction ---------------------------------------
    if (pmb->block_size.nx2 > 1) {
      int nc2 = cellbounds.ncellsj(entire);
      c1j = ParArrayND<Real>(PARARRAY_TEMP, nc2);
      c2j = ParArrayND<Real>(PARARRAY_TEMP, nc2);
      c3j = ParArrayND<Real>(PARARRAY_TEMP, nc2);
      c4j = ParArrayND<Real>(PARARRAY_TEMP, nc2);
      c5j = ParArrayND<Real>(PARARRAY_TEMP, nc2);
      c6j = ParArrayND<Real>(PARARRAY_TEMP, nc2);
      hplus_ratio_j = ParArrayND<Real>(PARARRAY_TEMP, nc2);
      hminus_ratio_j = ParArrayND<Real>(PARARRAY_TEMP, nc2);

      // zero-curvature PPM limiter does not depend on mesh uniformity:
      for (int j = (cellbounds.js(interior)) - 1; j <= (cellbounds.je(interior)) + 1;
           ++j) {
        // h_plus = 3.0;
        // h_minus = 3.0;
        // Ratios are = 2 for Cartesian coords, as in the original PPM limiter's
        // overshoot conditions
        hplus_ratio_j(j) = 2.0;
        hminus_ratio_j(j) = 2.0;
      }
      // 4th order reconstruction weights along Cartesian-like x2 w/ uniform spacing
      if (uniform[X2DIR]) {
#pragma omp simd
        for (int j = cellbounds.js(entire); j <= cellbounds.je(entire); ++j) {
          c1j(j) = 0.5;
          c2j(j) = 0.5;
          c3j(j) = 0.5;
          c4j(j) = 0.5;
          c5j(j) = 1.0 / 6.0;
          c6j(j) = -1.0 / 6.0;
        }
      } else { // coeffcients along Cartesian-like x2 with nonuniform mesh spacing
#pragma omp simd
        for (int j = pmb->cellbounds.js(entire) + 2; j <= pmb->cellbounds.je(entire) - 1;
             ++j) {
          Real dx_jm1 = coords.dx2f(j - 1);
          Real dx_j = coords.dx2f(j);
          Real dx_jp1 = coords.dx2f(j + 1);
          Real qe = dx_j / (dx_jm1 + dx_j + dx_jp1); // Outermost coeff in CW eq 1.7
          c1j(j) =
              qe * (2.0 * dx_jm1 + dx_j) / (dx_jp1 + dx_j); // First term in CW eq 1.7
          c2j(j) =
              qe * (2.0 * dx_jp1 + dx_j) / (dx_jm1 + dx_j); // Second term in CW eq 1.7

          if (j > pmb->cellbounds.js(entire) +
                      1) { // c3-c6 are not computed in first iteration
            Real dx_jm2 = coords.dx2f(j - 2);
            Real qa = dx_jm2 + dx_jm1 + dx_j + dx_jp1;
            Real qb = dx_jm1 / (dx_jm1 + dx_j);
            Real qc = (dx_jm2 + dx_jm1) / (2.0 * dx_jm1 + dx_j);
            Real qd = (dx_jp1 + dx_j) / (2.0 * dx_j + dx_jm1);
            qb = qb + 2.0 * dx_j * qb / qa * (qc - qd);
            c3j(j) = 1.0 - qb;
            c4j(j) = qb;
            c5j(j) = dx_j / qa * qd;
            c6j(j) = -dx_jm1 / qa * qc;
          }
        }
      } // end nonuniform Cartesian-like
    }   // end 2D or 3D

    // Precompute PPM coefficients in x3-direction
    if (pmb->block_size.nx3 > 1) {
      int nc3 = cellbounds.ncellsk(entire);
      c1k = ParArrayND<Real>(PARARRAY_TEMP, nc3);
      c2k = ParArrayND<Real>(PARARRAY_TEMP, nc3);
      c3k = ParArrayND<Real>(PARARRAY_TEMP, nc3);
      c4k = ParArrayND<Real>(PARARRAY_TEMP, nc3);
      c5k = ParArrayND<Real>(PARARRAY_TEMP, nc3);
      c6k = ParArrayND<Real>(PARARRAY_TEMP, nc3);
      hplus_ratio_k = ParArrayND<Real>(PARARRAY_TEMP, nc3);
      hminus_ratio_k = ParArrayND<Real>(PARARRAY_TEMP, nc3);

      // reconstruction coeffiencients in x3, Cartesian-like coordinate:
      if (uniform[X3DIR]) { // uniform spacing
#pragma omp simd
        for (int k = cellbounds.ks(entire); k <= cellbounds.ke(entire); ++k) {
          c1k(k) = 0.5;
          c2k(k) = 0.5;
          c3k(k) = 0.5;
          c4k(k) = 0.5;
          c5k(k) = 1.0 / 6.0;
          c6k(k) = -1.0 / 6.0;
        }

      } else { // nonuniform spacing
#pragma omp simd
        for (int k = pmb->cellbounds.ks(entire) + 2; k <= pmb->cellbounds.ke(entire) - 1;
             ++k) {
          Real dx_km1 = coords.dx3f(k - 1);
          Real dx_k = coords.dx3f(k);
          Real dx_kp1 = coords.dx3f(k + 1);
          Real qe = dx_k / (dx_km1 + dx_k + dx_kp1); // Outermost coeff in CW eq 1.7
          c1k(k) =
              qe * (2.0 * dx_km1 + dx_k) / (dx_kp1 + dx_k); // First term in CW eq 1.7
          c2k(k) =
              qe * (2.0 * dx_kp1 + dx_k) / (dx_km1 + dx_k); // Second term in CW eq 1.7

          if (k > pmb->cellbounds.ks(entire) +
                      1) { // c3-c6 are not computed in first iteration
            Real dx_km2 = coords.dx3f(k - 2);
            Real qa = dx_km2 + dx_km1 + dx_k + dx_kp1;
            Real qb = dx_km1 / (dx_km1 + dx_k);
            Real qc = (dx_km2 + dx_km1) / (2.0 * dx_km1 + dx_k);
            Real qd = (dx_kp1 + dx_k) / (2.0 * dx_k + dx_km1);
            qb = qb + 2.0 * dx_k * qb / qa * (qc - qd);
            c3k(k) = 1.0 - qb;
            c4k(k) = qb;
            c5k(k) = dx_k / qa * qd;
            c6k(k) = -dx_km1 / qa * qc;
          }
        }
        // Compute geometric factors for x3 limiter (Mignone eq 48)
        // (no curvilinear corrections in x3)
        for (int k = (pmb->cellbounds.ks(interior)) - 1;
             k <= (pmb->cellbounds.ke(interior)) + 1; ++k) {
          // h_plus = 3.0;
          // h_minus = 3.0;
          // Ratios are both = 2 for Cartesian and all curviliniear coords
          hplus_ratio_k(k) = 2.0;
          hminus_ratio_k(k) = 2.0;
        }
      }
    }
    for (int i = 0; i < kNrows; ++i) {
      delete[] beta[i];
    }
    delete[] beta;
  } // end "if PPM or full 4th order spatial integrator"
}

} // namespace parthenon
