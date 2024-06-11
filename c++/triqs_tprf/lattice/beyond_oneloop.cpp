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
  //if (2 *std::abs(wnval.index()) > Wwm.last_index())
  //  TRIQS_RUNTIME_ERROR << "gamma_3pnt: interpolating outside the Matsubara mesh of W. Please define W on a larger mesh (at least twice the inner Fermionic mesh).\n";
  //if (std::abs(wnpval.index()) + std::abs(wnval.index()) > wmesh_f.last_index())
  //  TRIQS_RUNTIME_ERROR << "gamma_3pnt: interpolating outside the Matsubara mesh of G. Please define G on a larger mesh (at least thrice the inner Fermionic mesh).\n";

  auto kmesh = std::get<1>(g_wk.mesh());
  auto _  = all_t{};

  std::complex<double> gamma;
  gamma = 0.0;

  for (auto kpp : kmesh){
    auto kppval = mesh::brzone::value_t{kpp};
    triqs::mesh::brzone::value_t kmkppval = kval - kppval;
    triqs::mesh::brzone::value_t kpmkpkppval = kpval - kval + kppval;

    // Interpolating w and k at the same time gives incorrect result for some reason?
    auto W_w =  W_wk(_, kmkppval);
    auto g1_w = g_wk(_,kppval);
    auto g2_w = g_wk(_,kpmkpkppval);

    g_w_t integrant_w(wmesh_f, g_wk.target_shape());
    for (auto wpp : wmesh_f) {
      auto wnppval = mesh::imfreq::value_t{wpp};
      triqs::mesh::imfreq::value_t wnmwnppval = wnval - wnppval;
      triqs::mesh::imfreq::value_t wnpmwnpwnpp = wnpval - wnval + wnppval;
      integrant_w[wpp] = - W_w(wnmwnppval)(0,0,0,0) * g1_w(wnppval) * g2_w(wnpmwnpwnpp);
    }
    auto integrant_t = make_gf_from_fourier(integrant_w);

    gamma += integrant_t(0.0)(0,0);
  }

  gamma = gamma / (kmesh.size());
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
  //if (std::abs(wnval.index()) + wmesh_f.last_index() > Wwm.last_index())
  //  TRIQS_RUNTIME_ERROR << "gamma_3pnt: interpolating outside the Matsubara mesh of W. Please define W on a larger mesh (at least twice the inner Fermionic mesh).\n";
  //if (std::abs(wnpval.index()) + std::abs(wnval.index()) + wmesh_f.last_index() > wmesh_f.last_index())
  //  TRIQS_RUNTIME_ERROR << "gamma_3pnt: interpolating outside the Matsubara mesh of G. Please define G on a larger mesh (at least thrice the inner Fermionic mesh).\n";

  g_w_t integrant_w(wmesh_f, g_w.target_shape());
  integrant_w() = 0.0;
  for (auto wpp : wmesh_f) {
    auto wnppval = mesh::imfreq::value_t{wpp};
    integrant_w[wpp] = - W_w(wnval-wnppval)(0,0,0,0) * g_w(wnppval) * g_w(wnpval - wnval + wnppval);
  }
  auto integrant_t = make_gf_from_fourier(integrant_w);

  auto gamma = integrant_t(0.0)(0,0);
  return gamma;
  }

  chi0_t gamma_3pnt(chi_w_cvt W_w, g_w_cvt g_w, mesh::imfreq wmesh_f) {
    chi0_t gamma_wwp({wmesh_f, wmesh_f}, W_w.target_shape());
    gamma_wwp() = 0.0;

    auto arr = mpi_view(gamma_wwp.mesh());
    #pragma omp parallel for
    for (int idx = 0; idx < arr.size(); idx++) {
      auto &[w, wp] = arr[idx];
      gamma_wwp[w,wp](0,0,0,0) += gamma_3pnt(w.value(), wp.value(), W_w, g_w, g_w.mesh());
    }

    gamma_wwp = mpi::all_reduce(gamma_wwp);
    return gamma_wwp;
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

  //if (std::abs(wnval.index()) + wmesh_f.last_index() > Wwm.last_index())
  //  TRIQS_RUNTIME_ERROR << "chiA_4pnt: interpolating outside the Matsubara mesh of W. Please define W on a larger mesh (at least twice the inner Fermionic mesh).\n";
  //if (std::abs(wnpval.index()) + std::abs(wnval.index()) + wmesh_f.last_index() > gwm.last_index())
  //  TRIQS_RUNTIME_ERROR << "chiA_4pnt: interpolating outside the Matsubara mesh of G. Please define G on a larger mesh (at least thrice the inner Fermionic mesh).\n";

  g_w_t integrant_w(wmesh_f, g_w.target_shape());
  integrant_w() = 0.0;
  for (auto wpp : wmesh_f) {
    auto wnppval = mesh::imfreq::value_t{wpp};
    integrant_w[wpp] = W_w(wnpval-wnppval)(0,0,0,0) * W_w(wnval - wnppval)(0,0,0,0) * g_w(wnpval + wnval - wnppval) * g_w(-wnppval);
  }
  auto integrant_t = make_gf_from_fourier(integrant_w);

  auto chi = integrant_t(0.0)(0,0);

  return chi;
  }


  chi0_t chiA_4pnt(chi_w_cvt W_w, g_w_cvt g_w, mesh::imfreq wmesh_f) {
    chi0_t chi_wwp({wmesh_f, wmesh_f}, W_w.target_shape());
    chi_wwp() = 0.0;

    auto arr = mpi_view(chi_wwp.mesh());
    #pragma omp parallel for
    for (int idx = 0; idx < arr.size(); idx++) {
      auto &[w, wp] = arr[idx];
      chi_wwp[w,wp](0,0,0,0) += chiA_4pnt(w.value(), wp.value(), W_w, g_w, g_w.mesh());
    }

    chi_wwp = mpi::all_reduce(chi_wwp);
    return chi_wwp;
  }



  
  std::complex<double> chiB_4pnt(mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_w_cvt W_w, g_w_cvt g_g_w, mesh::imfreq wmesh_f) {

  int nb = g_g_w.target().shape()[0];
  auto Wwm = W_w.mesh();
  auto gwm = g_g_w.mesh();

  if (Wwm.beta() != gwm.beta() || wmesh_f.beta() != gwm.beta())
    TRIQS_RUNTIME_ERROR << "chiB_4pnt: inverse temperatures are not the same.\n";
  if (Wwm.statistic() != Boson || gwm.statistic() != Fermion || wmesh_f.statistic() != Fermion)
    TRIQS_RUNTIME_ERROR << "chiB_4pnt: statistics are incorrect.\n";
  if (nb != 1)
    TRIQS_RUNTIME_ERROR << "chiB_4pnt: not implemented for multiorbital systems.\n";

  //if (std::abs(wnval.index()) + wmesh_f.last_index() > Wwm.last_index())
  //  TRIQS_RUNTIME_ERROR << "chiB_4pnt: interpolating outside the Matsubara mesh of W. Please define W on a larger mesh (at least twice the inner Fermionic mesh).\n";
  //if (std::abs(wnpval.index()) + std::abs(wnval.index()) + wmesh_f.last_index() > gwm.last_index())
  //  TRIQS_RUNTIME_ERROR << "chiB_4pnt: interpolating outside the Matsubara mesh of G. Please define G on a larger mesh (at least thrice the inner Fermionic mesh).\n";

  g_w_t integrant_w(wmesh_f, g_g_w.target_shape());
  integrant_w() = 0.0;
  for (auto wpp : wmesh_f) {
    auto wnppval = mesh::imfreq::value_t{wpp};
    integrant_w[wpp] = W_w(wnpval-wnppval)(0,0,0,0) * W_w(wnval - wnppval)(0,0,0,0) * g_g_w(wnppval);
  }
  auto integrant_t = make_gf_from_fourier(integrant_w);

  auto chi = integrant_t(0.0)(0,0);
  return chi;
  }

  chi0_t chiB_4pnt(chi_w_cvt W_w, g_wk_cvt g_wk, mesh::imfreq wmesh_f) {

    g_w_t g_g_w(std::get<0>(g_wk.mesh()), g_wk.target_shape());
    g_g_w() = 0.0;
    auto Nk = std::get<1>(g_wk.mesh()).size();
 
    auto arr_g = mpi_view(g_wk.mesh());
    #pragma omp parallel for
    for (int idx = 0; idx < arr_g.size(); idx++) {
      auto &[w,k] = arr_g[idx];
      g_g_w[w] += g_wk[w,k] * g_wk[-w,-k] / Nk;
    }
    g_g_w = mpi::all_reduce(g_g_w);


    chi0_t chi_wwp({wmesh_f, wmesh_f}, W_w.target_shape());
    chi_wwp() = 0.0;

    auto arr = mpi_view(chi_wwp.mesh());
    #pragma omp parallel for
    for (int idx = 0; idx < arr.size(); idx++) {
      auto &[w, wp] = arr[idx];
      chi_wwp[w,wp](0,0,0,0) += chiB_4pnt(w.value(), wp.value(), W_w, g_g_w, std::get<0>(g_wk.mesh()));
    }

    chi_wwp = mpi::all_reduce(chi_wwp);
    return chi_wwp;
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


  chi0_t sc_kernel(chi_w_cvt W_w, g_wk_cvt g_wk, g_w_cvt sigma_w, mesh::imfreq wmesh_f,
                   bool oneloop_kernel=true, bool gamma_kernel=true, bool sigma_kernel=true,
                   bool chiA_kernel=true, bool chiB_kernel=true) {

  auto [g_wmesh, g_kmesh] = g_wk.mesh();
  auto Nk = g_kmesh.size();
  g_w_t g_w(g_wmesh, g_wk.target_shape());
  g_w() = 0.0;
  g_w_t g_g_w(g_wmesh, g_wk.target_shape());
  g_g_w() = 0.0;
  g_w_t g_gn_gn_w(g_wmesh, g_wk.target_shape());
  g_gn_gn_w() = 0.0;
  g_w_t g_g_gn_w(g_wmesh, g_wk.target_shape());
  g_g_gn_w() = 0.0;
 
  auto arr_g = mpi_view(g_wk.mesh());
  #pragma omp parallel for
  for (int idx = 0; idx < arr_g.size(); idx++) {
    auto &[w,k] = arr_g[idx];
    g_w[w] += g_wk[w,k] / Nk;
    g_g_w[w] += g_wk[w,k] * g_wk[-w,-k] / Nk;
    g_gn_gn_w[w] += g_wk[w,k] * g_wk[-w,-k] * g_wk[-w,-k] / Nk;
    g_g_gn_w[w] += g_wk[w,k] * g_wk[w,k] * g_wk[-w,-k] / Nk;
  }
  g_w = mpi::all_reduce(g_w);
  g_g_w = mpi::all_reduce(g_g_w);
  g_gn_gn_w = mpi::all_reduce(g_gn_gn_w);
  g_g_gn_w = mpi::all_reduce(g_g_gn_w);


  chi0_t gamma_wwp, chiA_wwp, chiB_wwp;
  if (gamma_kernel)
    gamma_wwp = gamma_3pnt(W_w, g_w, wmesh_f);
  if (chiA_kernel)
    chiA_wwp = chiA_4pnt(W_w, g_w, wmesh_f);
  if (chiB_kernel)
    chiB_wwp = chiB_4pnt(W_w, g_wk, wmesh_f);


  chi0_t kernel_wwp({wmesh_f, wmesh_f}, W_w.target_shape());
  kernel_wwp() = 0.0;

  auto arr = mpi_view(kernel_wwp.mesh());
  #pragma omp parallel for
  for (int idx = 0; idx < arr.size(); idx++) {
    auto &[w, wp] = arr[idx];

    if (oneloop_kernel) {
      kernel_wwp[w,wp] += - W_w(w-wp) * g_g_w(wp)(0,0);
    }
    if (gamma_kernel) {
      kernel_wwp[w,wp] += - W_w(w-wp) * (gamma_wwp(w,wp) + gamma_wwp(-w,-wp)) * g_g_w(wp)(0,0);
    }
    if (sigma_kernel) {
      kernel_wwp[w,wp] += - W_w(w-wp) * sigma_w(-wp)(0,0) * g_gn_gn_w(wp)(0,0);
      kernel_wwp[w,wp] += - W_w(w-wp) * sigma_w(wp)(0,0) * g_g_gn_w(wp)(0,0);
    }
    if (chiA_kernel) {
      kernel_wwp[w,wp] += chiA_wwp(w,wp) * g_g_w(wp)(0,0);
    }
    if (chiB_kernel) {
      kernel_wwp[w,wp] += chiB_wwp(w,wp) * g_g_w(wp)(0,0);
    }
  }

  kernel_wwp = mpi::all_reduce(kernel_wwp);
  return kernel_wwp;
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


  auto wmesh_f = delta_w.mesh();
  auto kernel_wwp = sc_kernel(W_w, g_wk, sigma_w, wmesh_f, oneloop_kernel, gamma_kernel, sigma_kernel, chiA_kernel, chiB_kernel);

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
      for (auto wnp : wmesh_f) {
        local_eigenval += delta_w[wn](0,0) * kernel_wwp[wn,wnp](0,0,0,0) * delta_w[wnp](0,0) / (beta * beta);
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
