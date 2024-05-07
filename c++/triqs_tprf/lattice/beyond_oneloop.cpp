/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2022, The Simons Foundation
 * Authors: H. U.R. Strand, Y. in 't Veld, M. Rösner
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <nda/nda.hpp>
#include <nda/linalg/eigenelements.hpp>

#include "gw.hpp"
#include "common.hpp"
#include "lattice_utility.hpp"
#include "../mpi.hpp"

// -- For parallell Fourier transform routines
#include "gf.hpp"
#include "chi_imtime.hpp"

namespace triqs_tprf {

  std::complex<double> gamma_3pnt(mesh::brzone::value_t kval, triqs::mesh::brzone::value_t kpval, mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_wk_cvt W_wk, g_wk_cvt g_wk) {

  int nb = g_wk.target().shape()[0];
  auto Wwm = std::get<0>(W_wk.mesh());
  auto gwm = std::get<0>(g_wk.mesh());

  if (Wwm.beta() != gwm.beta())
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: inverse temperatures are not the same.\n";
  if (Wwm.statistic() != Boson || gwm.statistic() != Fermion)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: statistics are incorrect.\n";
  if (std::get<1>(W_wk.mesh()) != std::get<1>(g_wk.mesh()))
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: k-space meshes are not the same.\n";
  if (nb != 1)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: not implemented for multiorbital systems.\n";

  auto maxind = gwm.last_index() - Wwm.last_index();
  if (maxind < 0) maxind = 0;
  if (std::abs(wnval.index()) > maxind)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: wnval ("+std::to_string(wnval.index())+") lies outside of possible interpolation range of G ("+std::to_string(maxind)+").\n";
  if (std::abs(wnpval.index()) > maxind)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: wnpval ("+std::to_string(wnpval.index())+") lies outside of possible interpolation range of G ("+std::to_string(maxind)+").\n";

  auto wmesh_b = std::get<0>(W_wk.mesh());
  auto qmesh = std::get<1>(W_wk.mesh());
  auto beta = wmesh_b.beta();

  std::complex<double> gamma;
  gamma = 0.0;

  for (auto q : qmesh){
    auto qval = mesh::brzone::value_t{q};
    triqs::mesh::brzone::value_t kmqval = kval - qval;
    triqs::mesh::brzone::value_t kpmqval = kpval - qval;

    for (auto wm : wmesh_b) {
      auto wmval = mesh::imfreq::value_t{wm};

      auto W = W_wk[wm,q](0,0,0,0);
      auto g_1 = g_wk(wnval-wmval, kmqval)(0,0);
      auto g_2 = g_wk(wnpval-wmval, kpmqval)(0,0);

      gamma -= W * g_1 * g_2 / (beta * qmesh.size());
    }
  }

  return gamma;
  }


  std::complex<double> gamma_3pnt(mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_w_cvt W_w, g_w_cvt g_w) {

  int nb = g_w.target().shape()[0];
  auto Wwm = W_w.mesh();
  auto gwm = g_w.mesh();

  if (Wwm.beta() != gwm.beta())
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: inverse temperatures are not the same.\n";
  if (Wwm.statistic() != Boson || gwm.statistic() != Fermion)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: statistics are incorrect.\n";
  if (nb != 1)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: not implemented for multiorbital systems.\n";

  auto maxind = gwm.last_index() - Wwm.last_index();
  if (maxind < 0) maxind = 0;
  if (std::abs(wnval.index()) > maxind)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: wnval ("+std::to_string(wnval.index())+") lies outside of possible interpolation range of G ("+std::to_string(maxind)+").\n";
  if (std::abs(wnpval.index()) > maxind)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: wnpval ("+std::to_string(wnpval.index())+") lies outside of possible interpolation range of G ("+std::to_string(maxind)+").\n";

  auto wmesh_b = W_w.mesh();
  auto beta = wmesh_b.beta();

  std::complex<double> gamma;
  gamma = 0.0;

  for (auto wm : wmesh_b) {
    auto wmval = mesh::imfreq::value_t{wm};

    auto W = W_w[wm](0,0,0,0);
    auto g_1 = g_w(wnval-wmval)(0,0);
    auto g_2 = g_w(wnpval-wmval)(0,0);

    gamma -= W * g_1 * g_2 / beta;
  }

  return gamma;
  }



  std::complex<double> sc_kernel(mesh::brzone::value_t kval, triqs::mesh::brzone::value_t kpval, mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_wk_cvt W_wk, g_wk_cvt g_wk, g_wk_cvt sigma_wk,
                                 bool oneloop_kernel=true, bool gamma_kernel=true, bool sigma_kernel=true) {

  auto wmesh_b = std::get<0>(W_wk.mesh());

  auto maxind = Wwm.last_index();
  if (std::abs(wnval.index() - wnpval.index()) > maxind)
    TRIQS_RUNTIME_ERROR << "sc_kernel: wnval - wnpval ("+std::to_string(wnval.index() - wnpval.index())+") lies outside of possible interpolation range of W ("+std::to_string(maxind)+").\n";

  triqs::mesh::brzone::value_t negkpval = - kpval;
  triqs::mesh::brzone::value_t kmkpval = kval - kpval;
  mesh::imfreq::value_t wnmwnpval = wnval - wnpval;

  auto W = W_wk(wnmwnpval, kmkpval)(0,0,0,0);
  auto g_pos = g_wk(wnpval, kpval)(0,0);
  auto g_neg = g_wk(-wnpval, negkpval)(0,0);

  std::complex<double> kernel;
  kernel = 0;
  if (gamma_kernel) {
    auto gamma_pos = gamma_3pnt(kval, kpval, wnval, wnpval, W_wk, g_wk);
    auto gamma_neg = gamma_3pnt(-kval, -kpval, -wnval, -wnpval, W_wk, g_wk);
    kernel -= W * g_pos * g_neg * (gamma_pos + gamma_neg);
  }
  
  if (sigma_kernel)
    kernel -= W * g_pos * (sigma_wk(wnpval, kpval)(0,0) * g_pos + g_neg * sigma_wk(-wnpval, negkpval)(0,0)) * g_neg;
  
  if (oneloop_kernel)
    kernel -= g_pos * g_neg * W;

  return kernel;
  }


  std::complex<double> sc_eigenvalue(g_wk_cvt delta_wk, chi_wk_cvt W_wk, g_wk_cvt g_wk, g_wk_cvt sigma_wk,
                                     bool oneloop_kernel=true, bool gamma_kernel=true, bool sigma_kernel=true) {

  int nb = g_wk.target().shape()[0];
  auto Wwm = std::get<0>(W_wk.mesh());
  auto gwm = std::get<0>(g_wk.mesh());

  if (Wwm.beta() != gwm.beta())
    TRIQS_RUNTIME_ERROR << "sc_eigenvalue: inverse temperatures are not the same.\n";
  if (Wwm.statistic() != Boson || gwm.statistic() != Fermion)
    TRIQS_RUNTIME_ERROR << "sc_eigenvalue: statistics are incorrect.\n";
  if (std::get<1>(W_wk.mesh()) != std::get<1>(g_wk.mesh()))
    TRIQS_RUNTIME_ERROR << "sc_eigenvalue: k-space meshes are not the same.\n";
  if (nb != 1)
    TRIQS_RUNTIME_ERROR << "sc_eigenvalue: not implemented for multiorbital systems.\n";

  auto wmesh_f = std::get<0>(delta_wk.mesh());
  auto kmesh = std::get<1>(delta_wk.mesh());
  auto beta = wmesh_f.beta();

  std::complex<double> eigenval = 0;
  std::complex<double> local_eigenval = 0;
  std::complex<double> norm = 0;
  std::complex<double> local_norm = 0;
  
  auto arr = mpi_view(kmesh);
 
  #pragma omp parallel private(local_eigenval, local_norm)
  {
    #pragma omp parallel for
    for (unsigned int idx = 0; idx < arr.size(); idx++) {
      auto &k  = arr[idx];
      auto kval = mesh::brzone::value_t{k};
      for (auto kp : kmesh){
        auto kpval = mesh::brzone::value_t{kp};
        for (auto wn : wmesh_f) {
          auto wnval = mesh::imfreq::value_t{wn};
          for (auto wnp : wmesh_f) {
            auto wnpval = mesh::imfreq::value_t{wnp};
            auto kernel = sc_kernel(kval, kpval, wnval, wnpval, W_wk, g_wk, sigma_wk, oneloop_kernel, gamma_kernel, sigma_kernel);
            local_eigenval += delta_wk[wn,k](0,0) * kernel * delta_wk[wnp,kp](0,0) / (beta * beta * kmesh.size() * kmesh.size());
          }
        }
      }
    }

    #pragma omp parallel for
    for (unsigned int idx = 0; idx < arr.size(); idx++) {
      auto &k  = arr[idx];
      for (auto wn : wmesh_f) {
        local_norm += delta_wk[wn,k](0,0) * delta_wk[wn,k](0,0) / (beta * kmesh.size());
      }
    }

    #pragma omp critical
    {
      eigenval += local_eigenval;
      norm += local_norm;
    }
  }

  eigenval = mpi::all_reduce(eigenval);
  norm = mpi::all_reduce(norm);

  eigenval = eigenval / norm;
  return eigenval;
  }

} // namespace triqs_tprf
