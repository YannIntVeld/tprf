# ----------------------------------------------------------------------

import itertools
import numpy as np

# ----------------------------------------------------------------------

from triqs_tprf.lattice import fourier_wk_to_wr
from triqs_tprf.lattice import fourier_wr_to_wk
from triqs_tprf.lattice import fourier_wr_to_tr
from triqs_tprf.lattice import fourier_tr_to_wr

from triqs_tprf.lattice import chi_wr_from_chi_tr
from triqs_tprf.lattice import chi_tr_from_chi_wr
from triqs_tprf.lattice import chi_wk_from_chi_wr
from triqs_tprf.lattice import chi_wr_from_chi_wk 

from triqs_tprf.tight_binding import TBLattice

from triqs_tprf.lattice import lattice_dyson_g0_wk
from triqs_tprf.lattice import lattice_dyson_g_wk
from triqs_tprf.lattice import split_into_dynamic_wk_and_constant_k

from triqs_tprf.gw import bubble_PI_wk
from triqs_tprf.gw import dynamical_screened_interaction_W
from triqs_tprf.gw import gw_sigma
from triqs_tprf.gw import g0w_sigma

from triqs.gf import Gf, MeshImFreq, Idx, MeshImTime, MeshBrillouinZone
from triqs.gf.mesh_product import MeshProduct
from triqs.lattice.lattice_tools import BrillouinZone, BravaisLattice

# ----------------------------------------------------------------------

def test_gw_sigma_functions():
    """ Goes through the steps of a one-shot G0W0 calculation on the Matsubara frequency axis.
    It compares certain limits of the result with the real-frequency GW implementation.
    
    Author: Yann in 't Veld (2022) """ 

    nw = 1000
    nk = 8
    norb = 1
    beta = 10.0
    V = 1.0
    mu = 0.0
    
    t = -1.0 * np.eye(norb)
    
    t_r = TBLattice(
        units = [(1, 0, 0)],
        hopping = {
            (+1,) : t,
            (-1,) : t,
            },
        orbital_positions = [(0,0,0)]*norb,
        )
    
    kmesh = t_r.get_kmesh(n_k=(nk, 1, 1))
    e_k = t_r.fourier(kmesh)
    
    kmesh = e_k.mesh
    wmesh = MeshImFreq(beta, 'Fermion', nw)
    g0_wk = lattice_dyson_g0_wk(mu=mu, e_k=e_k, mesh=wmesh)
    
    V_k = Gf(mesh=kmesh, target_shape=[norb]*4)
    V_k.data[:] = V
    
    print('--> pi_bubble')
    PI_wk = bubble_PI_wk(g0_wk)
    
    print('--> dynamical_screened_interaction_W')
    Wr_full_wk = dynamical_screened_interaction_W(PI_wk, V_k)
    Wr_dyn_wk, Wr_stat_k = split_into_dynamic_wk_and_constant_k(Wr_full_wk)

    print('--> gw_sigma')
    sigma_wk = gw_sigma(Wr_full_wk, g0_wk)

    print('--> test static and dynamic parts')
    sigma_dyn_wk = gw_sigma(Wr_dyn_wk, g0_wk)
    sigma_stat_k = gw_sigma(V_k, g0_wk)

    sigma_wk_ref = Gf(mesh=sigma_dyn_wk.mesh, target_shape=sigma_dyn_wk.target_shape)
    for w in wmesh:
        iw = w.linear_index
        sigma_wk_ref.data[iw,:] = sigma_dyn_wk.data[iw,:] + sigma_stat_k.data[:]
 
    diff = sigma_wk.data[:] - sigma_wk_ref.data[:]
    print(np.max(np.abs(np.real(diff))))
    print(np.max(np.abs(np.imag(diff))))

    np.testing.assert_array_almost_equal(sigma_wk.data[:], sigma_wk_ref.data[:])

    print('--> test fourier transforms')
    ntau = nw*6+1
    g0_wr = fourier_wk_to_wr(g0_wk)
    g0_tr = fourier_wr_to_tr(g0_wr, nt=ntau)
    Wr_wr = chi_wr_from_chi_wk(Wr_dyn_wk)
    Wr_tr = chi_tr_from_chi_wr(Wr_wr, ntau=ntau)
  
    sigma_tr = gw_sigma(Wr_tr, g0_tr)
    sigma_wr = fourier_tr_to_wr(sigma_tr, nw=nw)
    sigma_dyn_wk_ref = fourier_wr_to_wk(sigma_wr)

    
    diff = sigma_dyn_wk.data[:] - sigma_dyn_wk_ref.data[:]
    print(np.max(np.abs(np.real(diff))))
    print(np.max(np.abs(np.imag(diff))))

    np.testing.assert_array_almost_equal(sigma_dyn_wk.data[:], sigma_dyn_wk_ref.data[:])
    
    print('--> g0w_sigma') 
    sigma_k = gw_sigma(V_k, g0_wk)
    sigma_k_ref = g0w_sigma(mu=mu, beta=beta, e_k=e_k, v_k=V_k)

    diff = sigma_k.data[:] - sigma_k_ref.data[:]
    print(np.max(np.abs(np.real(diff))))
    print(np.max(np.abs(np.imag(diff))))

    np.testing.assert_array_almost_equal(sigma_k.real.data[:], sigma_k_ref.real.data[:])
    np.testing.assert_array_almost_equal(sigma_k.imag.data[:], sigma_k_ref.imag.data[:], decimal=3)
    
    print('--> lattice_dyson_g_wk')
    g_wk = lattice_dyson_g_wk(mu, e_k, sigma_wk)





def nF(ww, beta):
    return 0.5 - 0.5 * np.tanh(0.5 * beta * ww)

def nB(ww, beta):
    return 1.0 / np.expm1( beta * ww )

def AnaSigma(iw, beta, g2, wD, E):
    fact1 = (nB(wD, beta) + nF(E, beta)) / (iw + wD - E)
    fact2 = (nB(wD, beta) + 1.0 - nF(E, beta)) / (iw - wD - E)
    return g2 * (fact1 + fact2)


def test_gw_sigma_functions2():
    """ Tests the Matsubara frequency axis GW implementation by comparing to an analytic result.
    This result was found for a calculation on a single k-point, with a simple electron-phonon propagator.
    Author: Yann in 't Veld (2023) """ 

    mu = 0.5
    g2 = 0.4
    wD = 0.1
    beta = 300.0
    nw = 500

    # Construct kmesh with only Gamma point
    bl = BravaisLattice(units=[(1,0,0)], orbital_positions=[(0,0,0)])
    bz = BrillouinZone(bl)
    kmesh = MeshBrillouinZone(bz, np.diag(np.array([1, 1,1], dtype=int)))
    wmesh = MeshImFreq(beta, 'Fermion', nw)
    numesh = MeshImFreq(beta, 'Boson', nw)

    print('--> lattice_dyson_g0_wk')
    Enk = Gf(mesh=kmesh, target_shape=[1]*2)
    Enk.data[:] = 0.0
    g0_fk = lattice_dyson_g0_wk(mu, Enk, wmesh)

    print('--> bare interaction')
    I_phon_wk = Gf(mesh=MeshProduct(numesh, kmesh), target_shape=[1]*4)
    for nu in numesh:
        nuii = nu.linear_index
        nuval = nu.value
        I_phon_wk.data[nuii,:] = g2 * 2.0 * wD / (nuval**2.0 - wD**2.0)

    print('--> gw_sigma')
    sigma_wk = gw_sigma(I_phon_wk, g0_fk)

    sigma_ref_wk = Gf(mesh=sigma_wk.mesh, target_shape=sigma_wk.target_shape)
    for f in wmesh:
        fii = f.linear_index
        fval = f.value
        sigma_ref_wk.data[fii,:] = AnaSigma(fval, beta, g2, wD, 0.0-mu)

    np.testing.assert_array_almost_equal(sigma_wk.data[:], sigma_ref_wk.data[:], decimal=1e-6)






if __name__ == "__main__":
    test_gw_sigma_functions()
    test_gw_sigma_functions2()
