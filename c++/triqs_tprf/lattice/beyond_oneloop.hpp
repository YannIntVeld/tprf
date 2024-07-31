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
#pragma once

#include "../types.hpp"

namespace triqs_tprf {

// Local vertices
  std::tuple<g_w_t,g_w_t,g_w_t,g_w_t> localized_gfs_for_sc(g_wk_cvt g_wk);
  std::tuple<g_Dw_t,g_Dw_t,g_Dw_t,g_Dw_t> localized_gfs_for_sc(g_Dwk_cvt g_wk);

  std::complex<double> gamma_3pnt(mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_w_cvt W_w, g_w_cvt g_w, mesh::imfreq wmesh_f);
  chi0_t gamma_3pnt(chi_w_cvt W_w, g_w_cvt g_w, mesh::imfreq wmesh_f);
  std::complex<double> chiA_4pnt(mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_w_cvt W_w, g_w_cvt g_w, mesh::imfreq wmesh_f);
  chi0_t chiA_4pnt(chi_w_cvt W_w, g_w_cvt g_w, mesh::imfreq wmesh_f);
  std::complex<double> chiB_4pnt(mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_w_cvt W_w, g_w_cvt g_g_k, mesh::imfreq wmesh_f);
  chi0_t chiB_4pnt(chi_w_cvt W_w, g_w_cvt g_g_w, mesh::imfreq wmesh_f);

  chi0_t sc_kernel_oneloop(chi_w_cvt W_w, g_w_cvt g_g_w, mesh::imfreq wmesh_f);
  chi0_t sc_kernel_gamma(chi_w_cvt W_w, chi0_cvt gamma_wwp, g_w_cvt g_g_w, mesh::imfreq wmesh_f);
  chi0_t sc_kernel_sigma(chi_w_cvt W_w, g_w_cvt sigma_w, g_w_cvt g_gn_gn_w, g_w_cvt g_g_gn_w, mesh::imfreq wmesh_f);
  chi0_t sc_kernel_chiA(chi0_cvt chiA_wwp, g_w_cvt g_g_w, mesh::imfreq wmesh_f);
  chi0_t sc_kernel_chiB(chi0_cvt chiB_wwp, g_w_cvt g_g_w, mesh::imfreq wmesh_f);
  std::complex<double> sc_eigenvalue(g_w_cvt delta_w, chi0_cvt kernel_wwp);

// Non-local vertices
  std::complex<double> gamma_3pnt(mesh::brzone::value_t kval, triqs::mesh::brzone::value_t kpval, mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_wk_cvt W_wk, g_wk_cvt g_wk, mesh::imfreq wmesh_f);

  std::complex<double> sc_kernel(mesh::brzone::value_t kval, triqs::mesh::brzone::value_t kpval, mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_wk_cvt W_wk, g_wk_cvt g_wk, g_wk_cvt sigma_wk,
                                 mesh::imfreq wmesh_f, bool oneloop_kernel=true, bool gamma_kernel=true, bool sigma_kernel=true);
  
  std::complex<double> sc_eigenvalue(g_wk_cvt delta_wk, chi_wk_cvt W_wk, g_wk_cvt g_wk, g_wk_cvt sigma_wk, bool oneloop_kernel=true, bool gamma_kernel=true, bool sigma_kernel=true);

} // namespace triqs_tprf
