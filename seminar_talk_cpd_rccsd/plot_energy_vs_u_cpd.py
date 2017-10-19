import numpy as np
import time
import pickle
from pyscf.lib import logger
from pyscf import scf
from tcc.hubbard import hubbard_from_scf
from tcc.cc_solvers import (classic_solver, root_solver)
from tcc.rccsd import RCCSD_UNIT
from tcc.rccsd_cpd import RCCSD_CPD_LS_T_HUB


def run_scfs(N, us, filename):
    scf_energies = []
    scf_mos = []
    scf_mo_energies = []
    for idxu, u in enumerate(us):
        rhf = hubbard_from_scf(scf.RHF, N, N, u, 'y')
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
    N = 14
    us = np.linspace(2, 5.5, num=10)
    lambdas = [3, ] * 6 + [7, ] * 3 + [10, ]
    rankst = np.array([7, 14, 21, 28]).astype('int')

    results = np.array(us)

    # Run all scfs here so we will have same starting points for CC
    run_scfs(N, us, 'calculated/{}-site/scfs_different_u.p'.format(N))
    with open('calculated/{}-site/scfs_different_u.p'.format(N), 'rb') as fp:
        ref_scfs = pickle.load(fp)

    for idxr, rank in enumerate(rankst):
        energies = []
        converged = False
        # timb = time.process_time()
        for idxu, (u, curr_scf) in enumerate(zip(us, ref_scfs)):
            tim = time.process_time()
            rhf = hubbard_from_scf(scf.RHF, N, N, u, 'y')
            rhf.max_cycle = 1
            rhf.scf()
            e_scf, mo_coeff, mo_energy = curr_scf
            cc = RCCSD_CPD_LS_T_HUB(rhf, rankt=rank,
                                    mo_coeff=mo_coeff,
                                    mo_energy=mo_energy)
            if not converged:
                amps = None

            converged, energy, amps = classic_solver(
                cc, lam=lambdas[idxu], conv_tol_energy=1e-8,
                conv_tol_amps=1e-7, max_cycle=20000,
                verbose=logger.NOTE)
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
        header='U '+' '.join('R={}'.format(rr) for rr in rankst)
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
    N = 14
    rankst = np.array([7, 14, 21, 28]).astype('int')

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
    plt.ylim(-13, -6)
    plt.xlim(2, 4)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst] + ['Exact', ] + ['RCCSD', ],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/energy_vs_u_{}_sites.eps'.format(N))


def run_ccsd():

    N = 14
    us = np.linspace(1, 4, num=20)

    from tcc.cc_solvers import residual_diis_solver
    energies = []
    amps = None
    for idxu, u in enumerate(us):
        rhf = hubbard_from_scf(scf.RHF, N, N, u, 'y')
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
        np.column_stack(us, energies),
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
