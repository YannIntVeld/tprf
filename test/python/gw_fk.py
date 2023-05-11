# ----------------------------------------------------------------------

import itertools
import numpy as np

# ----------------------------------------------------------------------

from triqs_tprf.tight_binding import TBLattice

from triqs_tprf.lattice import lattice_dyson_g0_fk
from triqs_tprf.lattice import lattice_dyson_g_fk
from triqs_tprf.gw import lindhard_chi00
from triqs_tprf.gw import g0w_sigma
from triqs_tprf.gw import dynamical_screened_interaction_W

from triqs.gf import Gf, MeshReFreq, inverse
from triqs.gf.mesh_product import MeshProduct

# ----------------------------------------------------------------------

def test_gw_self_energy_real_freq():
    nw = 100
    wmin = -1000.0
    wmax = +1000.0
    nk = 5
    norb = 1
    beta = 10.0
    V = 10.0
    mu = 0.0
    delta = 0.1
    
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
    fmesh = MeshReFreq(wmin, wmax, nw)
    g_fk = lattice_dyson_g0_fk(mu=mu, e_k=e_k, mesh=fmesh, delta=delta)
    
    V_k = Gf(mesh=kmesh, target_shape=[norb]*4)
    V_k.data[:] = V
    
    V_fk = Gf(mesh=g_fk.mesh, target_shape=[norb]*4)
    V_fk.data[:] = V
    
    print('--> pi_bubble')
    PI_fk = lindhard_chi00(e_k=e_k, mesh=fmesh, beta=beta, mu=mu, delta=delta)
    
    print('--> screened_interaction_W')
    Wr_fk = dynamical_screened_interaction_W(PI_fk, V_k) 
    
    print('--> gw_self_energy')
    sigma_fk = g0w_sigma(mu=mu, beta=beta, e_k=e_k, W_fk=Wr_fk, v_k=V_k, delta=delta)
    sigma_k = g0w_sigma(mu=mu, beta=beta, e_k=e_k, v_k=V_k)
    
    # Check if sigma_fk and sigma_k are the same at w->inf
    np.testing.assert_array_almost_equal(sigma_fk.data[-1,:], sigma_k.data[:], decimal=4)
   

    print('--> gw_self_energy with a mask')
    kmask = np.zeros(len(kmesh), dtype=bool)
    kmask[:] = False
    kmask[0] = True
    fmask = np.zeros(len(fmesh), dtype=bool)
    fmask[:] = True
    sigma_fk_masked = g0w_sigma(mu=mu, beta=beta, e_k=e_k, W_fk=Wr_fk, v_k=V_k, delta=delta, outerkmesh=kmesh, kmask=kmask, fmask=fmask)

    np.testing.assert_array_almost_equal(sigma_fk.data[:,0,:], sigma_fk_masked.data[:,0,:], decimal=4)
    np.testing.assert_array_almost_equal(sigma_fk_masked.data[:,1:,:], np.zeros_like(sigma_fk_masked.data[:,1:,:]), decimal=10)


    print('--> gw_self_energy with outer kmesh')
    outerkmesh = t_r.get_kmesh((nk, 1, 1))
    kmask = [True]
    fmask = [True]
    sigma_fk_withOuterMesh = g0w_sigma(mu=mu, beta=beta, e_k=e_k, W_fk=Wr_fk, v_k=V_k, delta=delta, outerkmesh=outerkmesh, kmask=kmask, fmask=fmask)

    karr = np.array(list(kmesh.values()))
    kx_ind = np.where( np.isclose(karr[:,1], 0.0) )[0]

    assert np.allclose( karr[kx_ind,1], 0.0) 
    np.testing.assert_array_almost_equal(
        sigma_fk.data[:,kx_ind,:], sigma_fk_withOuterMesh.data)


    print('--> lattice_dyson_g_wk')
    g_fk = lattice_dyson_g_fk(mu, e_k, sigma_fk, delta)


if __name__ == "__main__":
    test_gw_self_energy_real_freq()
