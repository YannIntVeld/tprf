/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2019, The Simons Foundation and S. Käser
 * Authors: S. Käser, H. U.R. Strand
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
  /** Splits a rank 4 tensor-valued Green's function into dynamic and constant parts by tail fitting

    Splits a general rank 4 tensor-valued Green's function :math:`\chi_{abcd}(i\omega_n, \mathbf{k})` 
    into a dynamic and a constant part in Matsubara frequency space by fitting
    the high-frequency tail.
    
    .. math ::
        \chi_{abcd}(i\omega_n, \mathbf{k}) = 
            \chi^{(dyn)}_{abcd}(i\omega_n, \mathbf{k})
            + \chi^{(stat)}_{abcd}(\mathbf{k})

  @param chi_wk : general rank 4 tensor-valued Green's function :math:`\chi_{abcd}(i\omega_n, \mathbf{k})`. 
  @return Tuple of chi_dyn_wk, the dynamic part of chi :math:`\chi^{(dyn)}_{abcd}(i\omega_n, \mathbf{k})`, which converges to zero for :math:`\omega_n \rightarrow \infty`, and chi_const_k, the part of chi that is constant in Matsubara frequency space :math:`\chi^{(stat)}_{abcd}(\mathbf{k})`.
  */
  std::tuple<chi_wk_t, chi_k_t> split_into_dynamic_wk_and_constant_k(chi_wk_cvt chi_wk);

   /** Calculates the density-density terms of a static Coulomb interaction
   in the band basis from the full Coulomb tensor in the orbital basis.
   
     .. math ::
         V_{ij}(\mathbf{k}, \mathbf{k}') = 
             \sum_{ab} \psi^\dagger_{ai}(\mathbf{k})
                       \psi^\dagger_{bj}(\mathbf{k}')
                       \psi_{bj}(\mathbf{k})
                       \psi_{ai}(\mathbf{k}')
                       V_{ab}(\mathbf{k} - \mathbf{k})
 
     @param Coulomb tensor V_k in the orbital basis :math:`V_{ab}(\mathbf{q})`
     @param Eigenvector :math:`\psi_{ai}(\mathbf{k})`
     @return Density-density Coulomb interaction Vb_kkp in the band basis :math:`V_{ij}(\mathbf{k}, \mathbf{k}')`
  */
  g_kk_t densdens_V_orb_to_band_basis(chi_k_cvt V_k, e_k_cvt psi_k);

  /** Calculates the density of states (DOS) of an electronic dispersion using Gaussian broadening.
  The function evaluates
    
    .. math ::
        N_{i}(E) = \frac{1}{N_k} \sum_{k} \delta(\epsilon_{\mathbf{k}, i} - E)

  where the delta function is approximated by a Gaussian function and $\epsilon_{\mathbf{k}, i}$
  is the diagonalized matrix-valued dispersion relation $\epsilon_{\bar{a}b}(\mathbf{k})$.

  @param eps_k diagonalized lattice dispersion :math:`\epsilon_(\mathbf{k},i)`
  @param E energy at which to evaluate the density of states
  @param sigma broadening used for the Gaussian function
  @return the density of states
  */
  nda::array<std::complex<double>,1> gaussian_dos(eps_cvt eps_k, double E, double sigma);

  /** Calculates the Coulomb pseudo-potential from a static density-density Coulomb interaction
  in the band basis, using Gaussian broadening.
  The function evaluates
    
    .. math ::
        \mu_{pot} = \frac{1}{N(0)} \frac{1}{N_k^2} \sum_{k,k'}
           \sum_{i,j} \delta(\epsilon_{\mathbf{k}, i} - \mu)
                      \delta(\epsilon_{\mathbf{k'}, j} - \mu)
                      V_{ij}(\mathbf{k}, \mathbf{k'})

  where the delta function is approximated by a Gaussian function and $\epsilon_{\mathbf{k}, i}$
  is the diagonalized matrix-valued dispersion relation $\epsilon_{\bar{a}b}(\mathbf{k})$.

  @param eps_k diagonalized lattice dispersion :math:`\epsilon_(\mathbf{k},i)`
  @param mu the chemical potential :math:`\mu`
  @param sigma broadening used for the Gaussian function
  @param Vb_kkp Density-density Coulomb interaction in the band basis :math:`V_{ij}(\mathbf{k}, \mathbf{k}')`
  @return the Coulomb pseudo-potential :math:`\mu_{pot}`
  */
  std::complex<double> densdens_V_pseudopotential(eps_cvt eps_k, double mu, double sigma, g_kk_t Vb_kkp);

} // namespace triqs_tprf
