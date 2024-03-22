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

  array<std::complex<double>, 2> gamma_3pnt(mesh::brzone::value_t kval, triqs::mesh::brzone::value_t kpval, mesh::imfreq::value_t wnval, mesh::imfreq::value_t wnpval, chi_wk_cvt W_wk, g_wk_cvt g_wk) {

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

  auto wmesh_b = std::get<0>(W_wk.mesh());
  auto qmesh = std::get<1>(W_wk.mesh());
  auto beta = wmesh_b.beta();

  array<std::complex<double>, 2> gamma(nb ,nb);
  gamma() = 0.0;

  auto arr = mpi_view(qmesh);
  #pragma omp parallel for
  for (unsigned int idx = 0; idx < arr.size(); idx++) {
    auto &q  = arr[idx];
    auto qval = mesh::brzone::value_t{q};

    triqs::mesh::brzone::value_t kmqval = kval - qval;
    triqs::mesh::brzone::value_t kpmqval = kpval - qval;

    for (auto wm : wmesh_b) {
      auto wmval = mesh::imfreq::value_t{wm};

      auto W = W_wk[wm,q];
      auto g_1 = g_wk(wnval-wmval, kmqval);
      auto g_2 = g_wk(wnpval-wmval, kpmqval);

      for (int a : range(nb)) {
        gamma(a,a) = gamma(a,a) - W(a,a,a,a) * g_1(a,a) * g_2(a,a) / (beta * qmesh.size());
      }
    }
  }

  gamma = mpi::all_reduce(gamma);
  return gamma;
  }

} // namespace triqs_tprf
