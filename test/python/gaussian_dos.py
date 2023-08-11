# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------

from triqs.utility import mpi
from triqs.gf import Gf, MeshBrZone
from triqs.gf.mesh_product import MeshProduct
#from triqs_tprf.tight_binding import TBLattice
from triqs.lattice.lattice_tools import BrillouinZone, BravaisLattice

from triqs_tprf.lattice import gaussian_dos
from triqs_tprf.lattice import densdens_V_orb_to_band_basis
from triqs_tprf.lattice import densdens_V_pseudopotential

import scipy.integrate as integrate

# ----------------------------------------------------------------------

def backfold(kvec, kmesh):
    # get the q vector in internal units
    kvecInt = kvec @ np.linalg.inv(kmesh.domain.units)

    # back-fold if nececcary
    for ii in range(len(kvecInt)):
        if(kvecInt[ii] >= 0.5):
            kvecInt[ii] -= 1.0 
        if(kvecInt[ii] < -0.5):
            kvecInt[ii] += 1.0 

    # get back-folded vector in 1/ang
    kvecFolded = kvecInt @ kmesh.domain.units
    return kvecFolded

def get_analytical_pseudopotential_TF(a0, mstar, mu, eps):
    A = a0*a0
    N0 = A*mstar / (2.0*np.pi)
    e2 = 14.399
    kF = np.sqrt(2.0 * mstar * mu)

    I = lambda x : integrate.quad(lambda theta:1.0/(np.sin(theta) + x), 0.0, np.pi)[0]
    mu_pseudo = N0 * e2 / (A * eps * kF) * I(2.0 * np.pi**2.0 * e2 * N0 / (A * eps * kF))
    return mu_pseudo

def getDelta(EE, sigma):
    # sigma is broadening of delta fct
    return 1.0 / (sigma * np.sqrt(2.0 * np.pi)) \
    * np.exp(-0.5 * EE**2.0 / sigma**2.0)

def get_python_pseudopotential(E_k, mu, sigma, W_bands, N0):
    kmesh, _ = W_bands.mesh.components
    nb = W_bands.target_shape[0]

    EBMuVal = 0.0

    for k in kmesh:
        ki = k.linear_index
        for kp in kmesh:
            kpi = kp.linear_index
            for i in range(nb):
                for j in range(nb):
                    Ekk = E_k.data[ki, i]-mu
                    Ekp = E_k.data[kpi, j]-mu
                    d1 = getDelta(Ekk,sigma)
                    d2 = getDelta(Ekp,sigma)
                    EBMuVal += np.real(d1 * d2 * W_bands.data[ki,kpi,i,j])
    
    EBMuVal = EBMuVal / (N0*len(kmesh)**2.0)
    return EBMuVal


def test_flat_dos():
    print("Single band")
    a0 = 2.0
    nk = 50
    sigma = 0.2
    mstar = 0.4 * 0.1314
    mu = 1.0

    print("--> setup dispersion")
    bl = BravaisLattice(units=[(a0,0,0), (0,a0,0)], orbital_positions=[(0,0,0)])
    bz = BrillouinZone(bl)
    kmesh = MeshBrZone(bz, np.diag([nk, nk, 1]))
    eps_k = Gf(mesh=kmesh, target_shape=[1])
    for k in kmesh:
        kfolded = backfold(k.value, kmesh)
        knorm = np.linalg.norm(kfolded)
        eps_k[k] = knorm**2.0 / (2.0 * mstar)

    print('--> get dos')
    dos = gaussian_dos(eps_k, mu, sigma)
    print(dos)

    print('--> compare with analytic result')
    A = a0*a0
    dos_ref = A*mstar / (2.0*np.pi)
    print(dos_ref)
    assert np.isclose(dos[0], dos_ref, rtol=1e-2)

    print('--> setup Coulomb interaction')
    eps = 1.0
    e2 = 14.399

    W_q = Gf(mesh=kmesh, target_shape=[1]*4)
    W_q.data[:] = 0.0
    for k in kmesh:
        qfolded = backfold(k.value, kmesh)
        qnorm = np.linalg.norm(qfolded)

        invV = A * qnorm * eps / (2.0 * np.pi * e2)
        W_q[k] = 1.0 / (invV + 2.0*np.sum(dos))

    psi_k = Gf(mesh=kmesh, target_shape=[1]*2)
    psi_k.data[:] = 1.0

    Vb_kkp = densdens_V_orb_to_band_basis(W_q, psi_k)

    print('--> densdens_V_pseudopotential')
    mu_pseudo = densdens_V_pseudopotential(eps_k, mu, sigma, Vb_kkp)
    print("num:", mu_pseudo)

    mu_pseudo_ref = get_analytical_pseudopotential_TF(a0, mstar, mu, eps)
    print("ana:", mu_pseudo_ref)

    #mu_pseudo_ref2 = get_python_pseudopotential(eps_k, mu, sigma, Vb_kkp, np.sum(dos))
    #print("python:", mu_pseudo_ref2)


def test_multiband():
    print("Multi band")
    a0 = 2.0
    nk = 50
    sigma = 0.2
    mstar1 = 0.4 * 0.1314
    mstar2 = 0.2 * 0.1314
    mu = 1.0
    shift = 0.2
    norb = 2

    print("--> setup dispersion")
    bl = BravaisLattice(units=[(a0,0,0), (0,a0,0)], orbital_positions=[(0,0,0)*norb])
    bz = BrillouinZone(bl)
    kmesh = MeshBrZone(bz, np.diag([nk, nk, 1]))

    eps_k = Gf(mesh=kmesh, target_shape=[norb])
    eps_k.data[:] = 0.0
    for k in kmesh:
        ki = k.linear_index
        kfolded = backfold(k.value, kmesh)
        knorm = np.linalg.norm(kfolded)
        eps_k.data[ki,0] = knorm**2.0 / (2.0 * mstar1)
        eps_k.data[ki,1] = knorm**2.0 / (2.0 * mstar2) + shift

    print('--> get dos')
    dos = gaussian_dos(eps_k, mu, sigma)
    print(dos.real)

    print('--> compare with analytic result')
    A = a0*a0
    dos_ref1 = A*mstar1 / (2.0*np.pi)
    dos_ref2 = A*mstar2 / (2.0*np.pi)
    dos_ref = np.array([dos_ref1, dos_ref2])
    print(dos_ref)

    assert np.allclose(dos, dos_ref, rtol=1e-2)

    print('--> setup Coulomb interaction')
    eps = 1.0
    e2 = 14.399

    V_q = Gf(mesh=kmesh, target_shape=[norb]*4)
    V_q.data[:] = 0.0
    for k in kmesh:
        qfolded = backfold(k.value, kmesh)
        qnorm = np.linalg.norm(qfolded)
        if(np.isclose(qnorm,0.0)): continue

        ki = k.linear_index
        V_q.data[ki,0,0,0,0] = 2.0 * np.pi * e2 / (A * qnorm * eps)
        V_q.data[ki,1,1,1,1] = 2.0 * np.pi * e2 / (A * qnorm * eps)
        V_q.data[ki,0,0,1,1] = 2.0 * np.pi * e2 / (A * qnorm * eps)
        V_q.data[ki,1,1,0,0] = 2.0 * np.pi * e2 / (A * qnorm * eps)

    psi_k = Gf(mesh=kmesh, target_shape=[norb]*2)
    for k in kmesh:
        psi_k[k] = np.identity(norb)

    Vb_kkp = densdens_V_orb_to_band_basis(V_q, psi_k)

    print('--> densdens_V_pseudopotential')
    mu_pseudo = densdens_V_pseudopotential(eps_k, mu, sigma, Vb_kkp)
    print(mu_pseudo)


if __name__ == "__main__":
    test_flat_dos()
    test_multiband()

