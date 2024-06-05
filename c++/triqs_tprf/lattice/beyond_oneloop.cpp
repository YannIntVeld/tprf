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

  std::complex<double> gamma_3pnt(mesh::brzone::value_t kval, triqs::mesh::brzone::value_t kpval, mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_wk_cvt W_wk, g_wk_cvt g_wk, mesh::imfreq wmesh_f) {

  int nb = g_wk.target().shape()[0];
  auto Wwm = std::get<0>(W_wk.mesh());
  auto gwm = std::get<0>(g_wk.mesh());

  if (Wwm.beta() != gwm.beta() || wmesh_f.beta() != gwm.beta())
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: inverse temperatures are not the same.\n";
  if (Wwm.statistic() != Boson || gwm.statistic() != Fermion || wmesh_f.statistic() != Fermion)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: statistics are incorrect.\n";
  if (std::get<1>(W_wk.mesh()) != std::get<1>(g_wk.mesh()))
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: k-space meshes are not the same.\n";
  if (nb != 1)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: not implemented for multiorbital systems.\n";

  if (std::abs(wnval.index()) + wmesh_f.last_index() > Wwm.last_index())
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: interpolating outside the Matsubara mesh of W. Please define W on a larger mesh (at least twice the inner Fermionic mesh).\n";
  if (std::abs(wnpval.index()) + std::abs(wnval.index()) + wmesh_f.last_index() > gwm.last_index())
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: interpolating outside the Matsubara mesh of G. Please define G on a larger mesh (at least thrice the inner Fermionic mesh).\n";

  auto kmesh = std::get<1>(g_wk.mesh());
  auto beta = wmesh_f.beta();

  std::complex<double> gamma;
  gamma = 0.0;

  for (auto kpp : kmesh){
    auto kppval = mesh::brzone::value_t{kpp};
    triqs::mesh::brzone::value_t kmkppval = kval - kppval;
    triqs::mesh::brzone::value_t kpmkpkppval = kpval - kval + kppval;

    for (auto wnpp : wmesh_f) {
      auto wnppval = mesh::imfreq::value_t{wnpp};
      triqs::mesh::imfreq::value_t wnmwnppval = wnval - wnppval;
      triqs::mesh::imfreq::value_t wnpmwnpwnpp = wnpval - wnval + wnppval;

      auto W = W_wk(wnmwnppval,kmkppval)(0,0,0,0);
      auto g_1 = g_wk(wnppval, kppval)(0,0);
      auto g_2 = g_wk(wnpmwnpwnpp, kpmkpkppval)(0,0);

      gamma -= W * g_1 * g_2;
    }
  }

  gamma = gamma / (beta * kmesh.size());
  return gamma;
  }


  std::complex<double> gamma_3pnt(mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_w_cvt W_w, g_w_cvt g_w, mesh::imfreq wmesh_f) {

  int nb = g_w.target().shape()[0];
  auto Wwm = W_w.mesh();
  auto gwm = g_w.mesh();

  if (Wwm.beta() != gwm.beta() || wmesh_f.beta() != gwm.beta())
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: inverse temperatures are not the same.\n";
  if (Wwm.statistic() != Boson || gwm.statistic() != Fermion || wmesh_f.statistic() != Fermion)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: statistics are incorrect.\n";
  if (nb != 1)
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: not implemented for multiorbital systems.\n";

  if (std::abs(wnval.index()) + wmesh_f.last_index() > Wwm.last_index())
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: interpolating outside the Matsubara mesh of W. Please define W on a larger mesh (at least twice the inner Fermionic mesh).\n";
  if (std::abs(wnpval.index()) + std::abs(wnval.index()) + wmesh_f.last_index() > gwm.last_index())
    TRIQS_RUNTIME_ERROR << "gamma_3pnt: interpolating outside the Matsubara mesh of G. Please define G on a larger mesh (at least thrice the inner Fermionic mesh).\n";

  auto beta = wmesh_f.beta();

  std::complex<double> gamma;
  gamma = 0.0;

  for (auto wnpp : wmesh_f) {
    auto wnppval = mesh::imfreq::value_t{wnpp};

    auto W = W_w(wnval - wnppval)(0,0,0,0);
    auto g_1 = g_w(wnppval)(0,0);
    auto g_2 = g_w(wnpval - wnval + wnppval)(0,0);

    gamma -= W * g_1 * g_2 / beta;
  }

  return gamma;
  }


  std::complex<double> chiA_4pnt(mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_w_cvt W_w, g_w_cvt g_w, mesh::imfreq wmesh_f) {

  int nb = g_w.target().shape()[0];
  auto Wwm = W_w.mesh();
  auto gwm = g_w.mesh();

  if (Wwm.beta() != gwm.beta() || wmesh_f.beta() != gwm.beta())
    TRIQS_RUNTIME_ERROR << "chiA_4pnt: inverse temperatures are not the same.\n";
  if (Wwm.statistic() != Boson || gwm.statistic() != Fermion || wmesh_f.statistic() != Fermion)
    TRIQS_RUNTIME_ERROR << "chiA_4pnt: statistics are incorrect.\n";
  if (nb != 1)
    TRIQS_RUNTIME_ERROR << "chiA_4pnt: not implemented for multiorbital systems.\n";

  if (std::abs(wnval.index()) + wmesh_f.last_index() > Wwm.last_index())
    TRIQS_RUNTIME_ERROR << "chiA_4pnt: interpolating outside the Matsubara mesh of W. Please define W on a larger mesh (at least twice the inner Fermionic mesh).\n";
  if (std::abs(wnpval.index()) + std::abs(wnval.index()) + wmesh_f.last_index() > gwm.last_index())
    TRIQS_RUNTIME_ERROR << "chiA_4pnt: interpolating outside the Matsubara mesh of G. Please define G on a larger mesh (at least thrice the inner Fermionic mesh).\n";

  auto beta = wmesh_f.beta();

  std::complex<double> chi;
  chi = 0.0;

  for (auto wnpp : wmesh_f) {
    auto wnppval = mesh::imfreq::value_t{wnpp};

    auto W1 = W_w(wnpval - wnppval)(0,0,0,0);
    auto W2 = W_w( wnval - wnppval)(0,0,0,0);
    auto g_1 = g_w(wnpval + wnval - wnppval)(0,0);
    auto g_2 = g_w(-wnppval)(0,0);

    chi += W1 * W2 * g_1 * g_2 / beta;
  }

  return chi;
  }
  
  
  std::complex<double> chiB_4pnt(mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_w_cvt W_w, g_wk_cvt g_wk, mesh::imfreq wmesh_f) {

  int nb = g_wk.target().shape()[0];
  auto Wwm = W_w.mesh();
  auto gwm = std::get<0>(g_wk.mesh());

  if (Wwm.beta() != gwm.beta() || wmesh_f.beta() != gwm.beta())
    TRIQS_RUNTIME_ERROR << "chiB_4pnt: inverse temperatures are not the same.\n";
  if (Wwm.statistic() != Boson || gwm.statistic() != Fermion || wmesh_f.statistic() != Fermion)
    TRIQS_RUNTIME_ERROR << "chiB_4pnt: statistics are incorrect.\n";
  if (nb != 1)
    TRIQS_RUNTIME_ERROR << "chiB_4pnt: not implemented for multiorbital systems.\n";

  if (std::abs(wnval.index()) + wmesh_f.last_index() > Wwm.last_index())
    TRIQS_RUNTIME_ERROR << "chiB_4pnt: interpolating outside the Matsubara mesh of W. Please define W on a larger mesh (at least twice the inner Fermionic mesh).\n";
  if (std::abs(wnpval.index()) + std::abs(wnval.index()) + wmesh_f.last_index() > gwm.last_index())
    TRIQS_RUNTIME_ERROR << "chiB_4pnt: interpolating outside the Matsubara mesh of G. Please define G on a larger mesh (at least thrice the inner Fermionic mesh).\n";

  auto beta = wmesh_f.beta();
  auto g_kmesh = std::get<1>(g_wk.mesh());

  std::complex<double> chi;
  chi = 0.0;

  for (auto wnpp : wmesh_f) {
    auto wnppval = mesh::imfreq::value_t{wnpp};

    std::complex<double> g_g_product = 0.0;
    for (auto k : g_kmesh){
      //g_g_product += g_wk[wnpp,k](0,0) * g_wk[-wnpp,-k](0,0) / g_kmesh.size();
      triqs::mesh::brzone::value_t kval = k.value();
      triqs::mesh::brzone::value_t negkval = -k.value();
      g_g_product += g_wk(wnppval, kval)(0,0) * g_wk(-wnppval, negkval)(0,0) / g_kmesh.size();
    }

    auto W1 = W_w(wnpval - wnppval)(0,0,0,0);
    auto W2 = W_w( wnval - wnppval)(0,0,0,0);

    chi += W1 * W2 * g_g_product / beta;
  }

  return chi;
  }




  std::complex<double> sc_kernel(mesh::brzone::value_t kval, triqs::mesh::brzone::value_t kpval, mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_wk_cvt W_wk, g_wk_cvt g_wk, g_wk_cvt sigma_wk,
                                 mesh::imfreq wmesh_f, bool oneloop_kernel=true, bool gamma_kernel=true, bool sigma_kernel=true) {

  auto Wwm = std::get<0>(W_wk.mesh());
  if (std::abs(wnval.index() - wnpval.index()) > Wwm.last_index())
    TRIQS_RUNTIME_ERROR << "sc_kernel: interpolating outside the Matsubara mesh of W. Please define W on a larger mesh (at least twice the inner Fermionic mesh).\n";

  triqs::mesh::brzone::value_t negkpval = - kpval;
  triqs::mesh::brzone::value_t kmkpval = kval - kpval;
  mesh::imfreq::value_t wnmwnpval = wnval - wnpval;

  auto W = W_wk(wnmwnpval, kmkpval)(0,0,0,0);
  auto g_pos = g_wk(wnpval, kpval)(0,0);
  auto g_neg = g_wk(-wnpval, negkpval)(0,0);

  std::complex<double> kernel;
  kernel = 0;
  if (gamma_kernel) {
    auto gamma_pos = gamma_3pnt(kval, kpval, wnval, wnpval, W_wk, g_wk, wmesh_f);
    auto gamma_neg = gamma_3pnt(-kval, -kpval, -wnval, -wnpval, W_wk, g_wk, wmesh_f);
    kernel -= W * g_pos * g_neg * (gamma_pos + gamma_neg);
  }
  
  if (sigma_kernel)
    kernel -= W * g_pos * (sigma_wk(wnpval, kpval)(0,0) * g_pos + g_neg * sigma_wk(-wnpval, negkpval)(0,0)) * g_neg;
  
  if (oneloop_kernel)
    kernel -= g_pos * g_neg * W;

  return kernel;
  }

  std::complex<double> sc_kernel(mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_w_cvt W_w, g_wk_cvt g_wk, g_w_cvt sigma_w,
                                 mesh::imfreq wmesh_f, bool oneloop_kernel=true, bool gamma_kernel=true, bool sigma_kernel=true,
                                 bool chiA_kernel=true, bool chiB_kernel=true) {

  auto Wwm = W_w.mesh();
  if (std::abs(wnval.index() - wnpval.index()) > Wwm.last_index())
    TRIQS_RUNTIME_ERROR << "sc_kernel: interpolating outside the Matsubara mesh of W. Please define W on a larger mesh (at least twice the inner Fermionic mesh).\n";

  mesh::imfreq::value_t wnmwnpval = wnval - wnpval;
  auto W = W_w(wnmwnpval)(0,0,0,0);

  auto g_wmesh = std::get<0>(g_wk.mesh());
  auto g_kmesh = std::get<1>(g_wk.mesh());
  g_w_t g_w(g_wmesh, g_wk.target_shape());
  g_w() = 0.0;
  for (auto w : g_wmesh)
    for (auto k : g_kmesh)
        g_w[w] += g_wk[w,k] / g_kmesh.size();

  std::complex<double> g_g_product = 0.0;
  for (auto k : g_kmesh){
    triqs::mesh::brzone::value_t kval = mesh::brzone::value_t{k};
    triqs::mesh::brzone::value_t negkval = -kval;
    g_g_product += g_wk(wnpval,kval)(0,0) * g_wk(-wnpval,negkval)(0,0) / g_kmesh.size();
  }

  std::complex<double> kernel;
  kernel = 0;
  if (gamma_kernel) {
    auto gamma_pos = gamma_3pnt(wnval, wnpval, W_w, g_w, wmesh_f);
    auto gamma_neg = gamma_3pnt(-wnval, -wnpval, W_w, g_w, wmesh_f);
    kernel -= W * g_g_product * (gamma_pos + gamma_neg);
  }

  if (chiA_kernel) {
    auto chiA = chiA_4pnt(wnval, wnpval, W_w, g_w, wmesh_f);
    kernel += chiA * g_g_product;
  }

  if (chiB_kernel) {
    auto chiB = chiB_4pnt(wnval, wnpval, W_w, g_wk, wmesh_f);
    kernel += chiB * g_g_product;
  }

  if (sigma_kernel) {
    std::complex<double> g_gn_gn_product = 0.0;
    std::complex<double> g_g_gn_product = 0.0;
    for (auto k : g_kmesh){
      triqs::mesh::brzone::value_t kval = mesh::brzone::value_t{k};
      triqs::mesh::brzone::value_t negkval = -kval;
      g_gn_gn_product += g_wk(wnpval,kval)(0,0) * g_wk(-wnpval,negkval)(0,0) * g_wk(-wnpval,negkval)(0,0) / g_kmesh.size();
      g_g_gn_product += g_wk(wnpval,kval)(0,0) * g_wk(wnpval,kval)(0,0) * g_wk(-wnpval,negkval)(0,0) / g_kmesh.size();
    }

    kernel -= W * sigma_w(-wnpval)(0,0) * g_gn_gn_product + W * sigma_w(wnpval)(0,0) * g_g_gn_product;
  }
  
  if (oneloop_kernel)
    kernel -= g_g_product * W;

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
            auto kernel = sc_kernel(kval, kpval, wnval, wnpval, W_wk, g_wk, sigma_wk, wmesh_f, oneloop_kernel, gamma_kernel, sigma_kernel);
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


  std::complex<double> sc_eigenvalue(g_w_cvt delta_w, chi_w_cvt W_w, g_wk_cvt g_wk, g_w_cvt sigma_w,
                                     bool oneloop_kernel=true, bool gamma_kernel=true, bool sigma_kernel=true,
                                     bool chiA_kernel=true, bool chiB_kernel=true) {

  int nb = g_wk.target().shape()[0];
  auto Wwm = W_w.mesh();
  auto gwm = std::get<0>(g_wk.mesh());

  if (Wwm.beta() != gwm.beta())
    TRIQS_RUNTIME_ERROR << "sc_eigenvalue: inverse temperatures are not the same.\n";
  if (Wwm.statistic() != Boson || gwm.statistic() != Fermion)
    TRIQS_RUNTIME_ERROR << "sc_eigenvalue: statistics are incorrect.\n";
  if (nb != 1)
    TRIQS_RUNTIME_ERROR << "sc_eigenvalue: not implemented for multiorbital systems.\n";

  auto wmesh_f = delta_w.mesh();
  auto kmesh = std::get<1>(g_wk.mesh());
  auto beta = wmesh_f.beta();

  std::complex<double> eigenval = 0;
  std::complex<double> local_eigenval = 0;
  std::complex<double> norm = 0;
  std::complex<double> local_norm = 0;
  
  auto arr = mpi_view(wmesh_f);
 
  #pragma omp parallel private(local_eigenval, local_norm)
  {
    #pragma omp parallel for
    for (unsigned int idx = 0; idx < arr.size(); idx++) {
      auto &wn = arr[idx];
      auto wnval = mesh::imfreq::value_t{wn};
      for (auto wnp : wmesh_f) {
        auto wnpval = mesh::imfreq::value_t{wnp};
        auto kernel = sc_kernel(wnval, wnpval, W_w, g_wk, sigma_w, wmesh_f, oneloop_kernel, gamma_kernel, sigma_kernel, chiA_kernel, chiB_kernel);
        local_eigenval += delta_w[wn](0,0) * kernel * delta_w[wnp](0,0) / (beta * beta);
      }
    }

    #pragma omp parallel for
    for (unsigned int idx = 0; idx < arr.size(); idx++) {
      auto &wn = arr[idx];
      local_norm += delta_w[wn](0,0) * delta_w[wn](0,0) / (beta);
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
