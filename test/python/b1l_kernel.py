# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------

from triqs.gf import *
from triqs.lattice.lattice_tools import BrillouinZone, BravaisLattice

from triqs_tprf.lattice import lattice_dyson_g0_wk
from triqs_tprf.lattice import sc_kernel

# ----------------------------------------------------------------------


def test_local_sc_kernel():
    Eval = -1.0
    wval = 0.3
    gval = 0.3
    Uval = 1.0
    wcut = 10.0
    beta = 20.0

    print('--> construct meshes') 
    bl = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)], orbital_positions=[(0,0,0)])
    bz = BrillouinZone(bl)
    kmesh = MeshBrillouinZone(bz, [1, 1, 1])
    
    nw_matsubara = int(wcut * beta / (2.0 * np.pi) - 0.5)
    wmesh_f  = MeshImFreq(beta, 'Fermion', nw_matsubara*3)
    wmesh_b = MeshImFreq(beta, 'Boson', nw_matsubara*2)
  
    print("Fermionic mesh:", len(wmesh_f))
    print("  Bosonic mesh:", len(wmesh_b))

    woutermesh_f  = MeshImFreq(beta, 'Fermion', nw_matsubara)
    print("Outer Fermionic mesh:", len(woutermesh_f))

    print('--> construct Gf and W') 
    Enk = Gf(mesh=kmesh, target_shape=[1]*2)
    Enk.data[:] = Eval
    
    g0_wk = lattice_dyson_g0_wk(mu=0.0, e_k=Enk, mesh=wmesh_f)
    g0_w = g0_wk[:,Idx([0,0,0])]
   
    W_wk = Gf(mesh=MeshProduct(wmesh_b,kmesh), target_shape=[1]*4)
    for w in wmesh_b:
        W_wk.data[w.data_index,:] = Uval + gval * 2.0 * wval / ((w.value)**2.0 - wval**2.0)
    W_w = W_wk[:,Idx([0,0,0])]

    print('--> get normal self-energy')
    sigma_wk =g0_wk.copy()
    sigma_wk.data[:] = 0.0

    sigma_w = sigma_wk[:,Idx(0,0,0)]

    print('--> sc_kernel (non-local)') 
    kvec = [0.0, 0.0, 0.0]
    K_ref_wwp = Gf(mesh=MeshProduct(woutermesh_f, woutermesh_f), target_shape=[1]*2)
    for w in woutermesh_f:
        for wp in woutermesh_f:
            K_ref_wwp.data[w.data_index,wp.data_index,0,0] = sc_kernel(kvec, kvec, w.value, wp.value, W_wk, g0_wk, sigma_wk,
                                                                       woutermesh_f, True, True, True)
    
    print('--> sc_kernel (local)') 
    K_wwp = Gf(mesh=MeshProduct(woutermesh_f, woutermesh_f), target_shape=[1]*2)
    for w in woutermesh_f:
        for wp in woutermesh_f:
            K_wwp.data[w.data_index,wp.data_index,0,0] = sc_kernel(w.value, wp.value, W_w, g0_wk, sigma_w,
                                                                   woutermesh_f, True, True, True)

    np.testing.assert_array_almost_equal(K_ref_wwp.data[:], K_wwp.data[:])

if __name__ == "__main__":
    test_local_sc_kernel()
