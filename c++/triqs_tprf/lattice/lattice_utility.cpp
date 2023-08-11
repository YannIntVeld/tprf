/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2019, The Simons Foundation
 * Authors: H. U.R. Strand
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
#include "../mpi.hpp"

namespace triqs_tprf {

  std::tuple<chi_wk_t, chi_k_t> split_into_dynamic_wk_and_constant_k(chi_wk_cvt chi_wk) {

    auto _               = all_t{};
    auto &[wmesh, kmesh] = chi_wk.mesh();

    chi_wk_t chi_dyn_wk(chi_wk.mesh(), chi_wk.target_shape());
    chi_dyn_wk() = 0.0;
    chi_k_t chi_const_k(kmesh, chi_wk.target_shape());
    chi_const_k() = 0.0;

    for (auto const &k : kmesh) {
      auto chi_w = chi_wk[_, k];
      auto tail  = std::get<0>(fit_tail(chi_w));

      for (auto [a, b, c, d] : chi_wk.target_indices()) chi_const_k[k](a, b, c, d) = tail(0, a, b, c, d);

      for (auto const &w : wmesh) chi_dyn_wk[w, k] = chi_wk[w, k] - chi_const_k[k];
    }

    return {chi_dyn_wk, chi_const_k};
  }

  g_kk_t densdens_V_orb_to_band_basis(chi_k_cvt V_k, e_k_cvt psi_k) {
    auto kmesh = V_k.mesh();
    g_kk_t Vb_kkp({kmesh, kmesh}, psi_k.target_shape());
    Vb_kkp() = 0.0;

    auto arr = mpi_view(Vb_kkp.mesh());
#pragma omp parallel for
    for (unsigned int idx = 0; idx < arr.size(); idx++) {
      auto &[k, kp] = arr(idx);

      auto Wq = V_k(k-kp);
      for (const auto &[i,j] : Vb_kkp.target_indices()) {
        for (const auto &[a,b] : Vb_kkp.target_indices()) {
          Vb_kkp[k,kp](i,j) += nda::conj(psi_k[k](a,i)) * nda::conj(psi_k[kp](b,j))
                             * psi_k[k](b,j) * psi_k[kp](a,i)
                             * Wq(a,a,b,b);
        }
      }
    }

    Vb_kkp = mpi::all_reduce(Vb_kkp);
    return Vb_kkp;
  }

  std::complex<double> gaussian(std::complex<double> E, double sigma) {
    return 1.0 / (sigma * sqrt(2.0*M_PI)) * exp(-0.5*E*E / (sigma*sigma));
  }

  nda::array<std::complex<double>,1> gaussian_dos(eps_cvt eps_k, double E, double sigma) {
    auto kmesh = eps_k.mesh();
    int nb     = eps_k.target().shape()[0];

    nda::array<std::complex<double>,1> DOS(nb);
    DOS() = 0.0;

    auto arr = mpi_view(kmesh);
    for (unsigned int idx = 0; idx < arr.size(); idx++) {
      auto &k = arr(idx);
      
      for (int i : range(nb)) {
        DOS(i) += gaussian(eps_k[k](i) - E, sigma);
      }
    }

    DOS = mpi::all_reduce(DOS);
    DOS /= kmesh.size();
    return DOS;
  }

  std::complex<double> densdens_V_pseudopotential(eps_cvt eps_k, double mu, double sigma, g_kk_t Vb_kkp) {
    auto N0_b = gaussian_dos(eps_k, mu, sigma);
    auto nb = N0_b.shape()[0];
    std::complex<double> N0 = 0.0;
    for (int i : range(nb))
        N0 += N0_b(i);

    auto kmesh = eps_k.mesh();

    std::complex<double> pseudo_mu = 0.0;

    auto arr = mpi_view(Vb_kkp.mesh());
    for (unsigned int idx = 0; idx < arr.size(); idx++) {
      auto &[k, kp] = arr(idx);
      for (const auto &[i,j] : Vb_kkp.target_indices()) {
        pseudo_mu += gaussian(eps_k[k](i) - mu, sigma)
                   * gaussian(eps_k[kp](j) - mu, sigma)
                   * Vb_kkp[k,kp](i,j);
      }
    }

    pseudo_mu = mpi::all_reduce(pseudo_mu);
    pseudo_mu /= N0 * kmesh.size() * kmesh.size();
    return pseudo_mu;
  }

} // namespace triqs_tprf
