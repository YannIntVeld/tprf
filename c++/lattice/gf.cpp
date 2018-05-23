/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2017, H. U.R. Strand
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

#include <triqs/arrays/linalg/det_and_inverse.hpp>
using triqs::arrays::inverse;

#include "common.hpp"
#include "gf.hpp"

namespace tprf {
  
// ----------------------------------------------------
// g

gk_iw_t g0k_from_ek(double mu, ek_vt ek, g_iw_t::mesh_t mesh) {

  auto I = make_unit_matrix<ek_vt::scalar_t>(ek.target_shape()[0]);
  gk_iw_t g0k = make_gf<gk_iw_t::mesh_t::var_t>({mesh, ek.mesh()}, ek.target());
  
  //for (auto const &k : ek.mesh()) {

#pragma omp parallel for 
  for (int idx = 0; idx < ek.mesh().size(); idx++) {
    auto iter = ek.mesh().begin(); iter += idx; auto k = *iter;
    
    for (auto const &w : mesh) g0k[w, k] = inverse((w + mu)*I - ek(k));
  }

  return g0k;
}

gk_iw_t gk_from_ek_sigma(double mu, ek_vt ek, g_iw_vt sigma) {

  auto mesh = sigma.mesh();
  auto I = make_unit_matrix<ek_vt::scalar_t>(ek.target_shape()[0]);
  gk_iw_t gk =
      make_gf<gk_iw_t::mesh_t::var_t>({mesh, ek.mesh()}, ek.target());

  //for (auto const &k : ek.mesh()) {

#pragma omp parallel for 
  for (int idx = 0; idx < ek.mesh().size(); idx++) {
    auto iter = ek.mesh().begin(); iter += idx; auto k = *iter;
    
    for (auto const &w : mesh) gk[w, k] = inverse((w + mu)*I - ek(k) - sigma[w]);
  }

  return gk;
}

gr_iw_t gr_from_gk(gk_iw_vt gwk) {

  auto _ = var_t{};
  auto target = gwk.target();

  auto [wmesh, kmesh] = gwk.mesh();
  auto rmesh = make_adjoint_mesh(kmesh);

  gr_iw_t gwr = make_gf<gr_iw_t::mesh_t::var_t>({wmesh, rmesh}, target);

  auto w0 = *wmesh.begin();
  void * p = _fourier_plan<0>(gf_const_view(gwk[w0, _]), gf_view(gwr[w0, _]));

#pragma omp parallel for 
  for (int idx = 0; idx < wmesh.size(); idx++) {
    auto iter = wmesh.begin(); iter += idx; auto w = *iter;

    auto gr = make_gf<cyclic_lattice>(rmesh, target);
    auto gk = make_gf<brillouin_zone>(kmesh, target);

#pragma omp critical
    gk = gwk[w, _];

    _fourier_with_plan<0>(gf_const_view(gk), gf_view(gr), p);

#pragma omp critical
    gwr[w, _] = gr;

  }

  _fourier_destroy_plan(p);

  return gwr;
}

gr_iw_t gr_from_gk_serial(gk_iw_vt gwk) {

  auto [wmesh, kmesh] = gwk.mesh();
  auto rmesh = make_adjoint_mesh(kmesh);

  gr_iw_t gwr = make_gf<gr_iw_t::mesh_t::var_t>({wmesh, rmesh}, gwk.target());

  auto _ = var_t{};
  for (auto const &w : wmesh) gwr[w, _]() = fourier(gwk[w, _]);

  return gwr;
}

gk_iw_t gk_from_gr(gr_iw_vt gwr) {

  auto _ = var_t{};
  auto target = gwr.target();

  auto [wmesh, rmesh] = gwr.mesh();
  auto kmesh = make_adjoint_mesh(rmesh);
  
  gk_iw_t gwk = make_gf<gk_iw_t::mesh_t::var_t>({wmesh, kmesh}, target);

  auto w0 = *wmesh.begin();
  void * p = _fourier_plan<0>(gf_const_view(gwr[w0, _]), gf_view(gwk[w0, _]));

#pragma omp parallel for 
  for (int idx = 0; idx < wmesh.size(); idx++) {
    auto iter = wmesh.begin(); iter += idx; auto w = *iter;

    auto gr = make_gf<cyclic_lattice>(rmesh, target);
    auto gk = make_gf<brillouin_zone>(kmesh, target);

#pragma omp critical
    gr = gwr[w, _];

    _fourier_with_plan<0>(gf_const_view(gr), gf_view(gk), p);

#pragma omp critical
    gwk[w, _] = gk;

  }

  _fourier_destroy_plan(p);
  
  return gwk;
}

gk_iw_t gk_from_gr_serial(gr_iw_vt gwr) {

  auto [wmesh, rmesh] = gwr.mesh();
  auto kmesh = make_adjoint_mesh(rmesh);
  
  gk_iw_t gwk = make_gf<gk_iw_t::mesh_t::var_t>({wmesh, kmesh}, gwr.target());

  auto _ = var_t{};
  for (auto const &w : wmesh) gwk[w, _]() = fourier(gwr[w, _]);
  
  return gwk;
}
  
// ----------------------------------------------------
// Transformations: Matsubara frequency <-> imaginary time

gr_tau_t grt_from_grw(gr_iw_vt grw, int ntau) {

  auto wmesh = std::get<0>(grw.mesh());
  auto rmesh = std::get<1>(grw.mesh());

  double beta = wmesh.domain().beta;

  int nw = wmesh.last_index() + 1;
  if( ntau <= 0 ) ntau = 4 * nw;

  gr_tau_t grt = make_gf<gr_tau_t::mesh_t::var_t>(
      {{beta, Fermion, ntau}, rmesh}, grw.target());

  auto _ = var_t{};

  auto nb = grw.target_shape()[0];

  auto known_moments_zero = array<dcomplex, 3>(2, nb, nb);
  known_moments_zero() = 0.;

  auto known_moments = array<dcomplex, 3>(2, nb, nb);
  known_moments(0, range(), range()) = 0.;
  known_moments(1, range(), range()) = make_unit_matrix<dcomplex>(nb);
  
  for (auto const &r : rmesh) {

    auto gw = make_gf<imfreq>({beta, Fermion, nw}, grw.target());
    auto gt = make_gf<imtime>({beta, Fermion, ntau}, grw.target());

    gw = grw[_, r];

    if(r.linear_index() == 0) {
      auto [tail, err] = fit_tail(gw);
      std::cout << "0\n";
      std::cout << tail(0, range(), range()) << "\n";
      std::cout << known_moments(0, range(), range()) << "\n";
      std::cout << "1\n";
      std::cout << tail(1, range(), range()) << "\n";
      std::cout << known_moments(1, range(), range()) << "\n";
      std::cout << "2\n";
      std::cout << tail(2, range(), range()) << "\n";
      std::cout << "3\n";
      std::cout << tail(3, range(), range()) << "\n";
    }

    if(r.linear_index() == 0) {
      auto [tail, err] = fit_tail(gw, known_moments);
      std::cout << "0\n";
      std::cout << tail(0, range(), range()) << "\n";
      std::cout << known_moments(0, range(), range()) << "\n";
      std::cout << "1\n";
      std::cout << tail(1, range(), range()) << "\n";
      std::cout << known_moments(1, range(), range()) << "\n";
      std::cout << "2\n";
      std::cout << tail(2, range(), range()) << "\n";
      std::cout << "3\n";
      std::cout << tail(3, range(), range()) << "\n";
    }
    
    gt() = fourier<0>(gw);
    
    //_fourier<0>(gf_const_view(gw), gf_view(gt));

    //gt = fourier(gw, known_moments);

    /*
    {
    if(r.linear_index() == 0)
      _fourier<0>(gf_const_view(gw), gf_view(gt), array_const_view<dcomplex, 3>(known_moments));
    else
      _fourier<0>(gf_const_view(gw), gf_view(gt), array_const_view<dcomplex, 3>(known_moments_zero));
    }
    */

    grt[_, r] = gt;
  }

  return grt;
}

} // namespace tprf
