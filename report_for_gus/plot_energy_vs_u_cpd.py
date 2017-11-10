"""
This script runs RCCSD_CPD calculation on a 1D Hubbard w/o PBC
for different ranks of the decomposition. The RCCSD_CPD code
is new (as of Oct 18 2017) with explicit symmetrization of CPD
amplitudes.

"""

import numpy as np
import time

import pickle
from pyscf.lib import logger
from pyscf import scf
from tcc.hubbard import hubbard_from_scf
from tcc.cc_solvers import (classic_solver, root_solver, step_solver)
from tcc.rccsd import RCCSD_UNIT
from tcc.rccsd_cpd import RCCSD_nCPD_LS_T_HUB
from tcc.rccsd_cpd import RCCSD_CPD_LS_T_HUB

# Set up script parameters
# USE_PBC = 'y'
# U_VALUES = np.linspace(1, 5.5, num=20)
# U_VALUES_REFERENCE = np.linspace(1, 4, num=20)
# N = 14
# RANKS_T = np.array([7, 14, 21, 28, ]).astype('int')
# l1 = [3, ] * 12 + [4, ] * 12 + [10, ] * 2
# LAMBDAS = [l1, l1, l1, l1]

USE_PBC = 'y'
U_VALUES = np.linspace(1, 10.0, num=20)
U_VALUES_REFERENCE = np.linspace(1, 10, num=20)
N = 10
RANKS_T = np.array([5, 8, 10, 11, 12]).astype('int')
# l1 = [3, ] * 9 + [7, ] * 8 + [10, ] * 3
l1 = [1, ] * 4 + [3, ] * 13 + [5, ] * 3
l2 = [1, ] * 4 + [3, ] * 10 + [5, ] * 6
l3 = [1, ] * 4 + [3, ] * 5 + [5, ] * 5 + [7, ] * 6
LAMBDAS = [l1, l2, l3, l3, l3]

# USE_PBC = 'y'
# U_VALUES = np.linspace(1, 14.0, num=20)
# U_VALUES_REFERENCE = np.linspace(1, 14, num=20)
# N = 6
# RANKS_T = np.array([3, 4, 5, 6, 7]).astype('int')

# l1 = [3, ] * 12 + [7, ] * 6 + [10, ] * 2
# l2 = [3, ] * 9 + [5, ] * 3 + [7, ] * 5 + [10, ] * 3
# LAMBDAS = [l1, l2, l1, l1, l1]


def run_scfs(N, us, filename):
    scf_energies = []
    scf_mos = []
    scf_mo_energies = []
    for idxu, u in enumerate(us):
        rhf = hubbard_from_scf(scf.RHF, N, N, u, USE_PBC)
        rhf.scf()
        scf_energies.append(rhf.e_tot)
        scf_mos.append(rhf.mo_coeff)
        scf_mo_energies.append(rhf.mo_energy)

    results = tuple(
        tuple([e_tot, mo_coeff, mo_energy])
        for e_tot, mo_coeff, mo_energy
        in zip(scf_energies, scf_mos, scf_mo_energies))

    with open(filename, 'wb') as fp:
        pickle.dump(results, fp)


def calc_energy_vs_u_cpd():
    """
    Plot energy of RCCSD-CPD for all corellation strengths
    """
    # Set up parameters of the script
    us = U_VALUES.copy()
    lambdas = LAMBDAS.copy()
    rankst = RANKS_T.copy()

    results = np.array(us)

    print('Running CC-CPD')
    # Run all scfs here so we will have same starting points for CC
    run_scfs(N, us, 'calculated/{}-site/scfs_different_u.p'.format(N))
    with open('calculated/{}-site/scfs_different_u.p'.format(N), 'rb') as fp:
        ref_scfs = pickle.load(fp)

    for idxr, rank in enumerate(rankst):
        energies = []
        converged = False
        # timb = time.process_time()
        amps = None
        for idxu, (u, curr_scf) in enumerate(zip(us, ref_scfs)):
            tim = time.process_time()
            rhf = hubbard_from_scf(scf.RHF, N, N, u, USE_PBC)
            rhf.max_cycle = 1
            rhf.scf()
            e_scf, mo_coeff, mo_energy = curr_scf
            cc = RCCSD_CPD_LS_T_HUB(rhf, rankt={'t2': rank},
                                    mo_coeff=mo_coeff,
                                    mo_energy=mo_energy)
            if not converged:
                amps = None

            converged, energy, amps = classic_solver(
                cc, lam=lambdas[idxr][idxu], conv_tol_energy=1e-8,
                conv_tol_amps=1e-7, max_cycle=40000,
                verbose=logger.NOTE)
            # converged, energy, amps = step_solver(
            #     cc, beta=0.7,  # (1 - 1. / lambdas[idxr][idxu]),
            #     conv_tol_energy=1e-8,
            #     conv_tol_amps=1e-7, max_cycle=20000,
            #     verbose=logger.NOTE)

            if np.isnan(energy):
                Warning(
                    'Warning: N = {}, U = {} '
                    'Rank = {} did not converge'.format(N, u, rank)
                )

            energies.append(energy + e_scf)
            elapsed = time.process_time() - tim
            print('Step {} out of {}, rank = {}, time: {}\n'.format(
                idxu + 1, len(us), rank, elapsed
            ))

        results = np.column_stack((results, energies))
        # elapsedb = time.process_time() - timb
        # print('Batch {} out of {}, rank = {}, time: {}'.format(
        #  0 + 1, len(rankst), rank, elapsedb))

    np.savetxt(
        'calculated/{}-site/energy_vs_u.txt'.format(N),
        results,
        header='U ' + ' '.join('R={}'.format(rr) for rr in rankst)
    )
    us, *energies_l = np.loadtxt(
        'calculated/{}-site/energy_vs_u.txt'.format(N), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(us, energies)
    plt.xlabel('$U$')
    plt.ylabel('$E$, H')
    plt.title('Energy behavior for different ranks')
    fig.show()


def plot_energy_vs_u_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    rankst = RANKS_T.copy()

    us, *energies_l = np.loadtxt(
        'calculated/{}-site/energy_vs_u.txt'.format(N), unpack=True)

    if len(energies_l) != len(rankst):
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(rankst)))
    fig, ax = plt.subplots()

    rankst = rankst  # Cut out some elements
    for idxr, (rank, color) in enumerate(zip(rankst, colors)):
        ax.plot(us, energies_l[idxr],
                color=color, marker=None)
    u_ref, e_ref = np.loadtxt(
        'reference/{}-site/Exact'.format(N), unpack=True)
    ax.plot(u_ref, e_ref,
            color='k', marker=None)

    u_cc, e_cc = np.loadtxt(
        'reference/{}-site/RCCSD.txt'.format(N), unpack=True)
    ax.plot(u_cc, e_cc,
            ':k', marker=None)

    plt.xlabel('$U / t$')
    plt.ylabel('$E$, H')
    plt.title('Energy behavior for different ranks, {} sites'.format(N))
    # plt.ylim(-14.5, -5.5)   # 14 sites
    # plt.xlim(1, 5.5)
    plt.ylim(-12, 5)   # 10 sites
    plt.xlim(1, 9)
    # plt.ylim(-12, 2)   # 6 sites
    # plt.xlim(1, 14)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst] + ['Exact', ] + ['RCCSD', ],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/energy_vs_u_{}_sites.eps'.format(N))


def run_ccsd():

    us = U_VALUES_REFERENCE.copy()

    from tcc.cc_solvers import residual_diis_solver
    energies = []
    amps = None
    for idxu, u in enumerate(us):
        rhf = hubbard_from_scf(scf.RHF, N, N, u, USE_PBC)
        rhf.scf()
        scf_energy = rhf.e_tot
        cc = RCCSD_UNIT(rhf)
        converged, energy, amps = residual_diis_solver(cc,
                                                       conv_tol_energy=1e-6,
                                                       lam=15, amps=amps,
                                                       max_cycle=2000)
        energies.append(scf_energy + energy)

    np.savetxt(
        'reference/{}-site/RCCSD.txt'.format(N),
        np.column_stack((us, energies)),
        header='U E_total'
    )
    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(us, energies)
    plt.xlabel('$U$')
    plt.ylabel('$E$, H')
    plt.title('Energy behavior for different ranks')
    fig.show()


if __name__ == '__main__':
    calc_energy_vs_u_cpd()
