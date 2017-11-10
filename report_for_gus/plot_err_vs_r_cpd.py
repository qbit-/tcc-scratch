import pickle
import numpy as np
import time
from pyscf.lib import logger
from pyscf import scf
from tcc.hubbard import hubbard_from_scf
from tcc.cc_solvers import (classic_solver, root_solver)
from tcc.rccsd import RCCSD_UNIT
from tcc.rccsd_cpd import RCCSD_nCPD_LS_T_HUB

# Set up parameters of the script
N = 18
U = 2
RANKS_T = np.round(N**np.linspace(0.2, 1.8, num=14)).astype('int')


def calc_solutions_diff_r_eq_cpd():
    """
    Plot error of RCCSD-CPD vs rank for weak corellation
    """
    rankst = RANKS_T.copy()

    # Run RHF calculations
    rhf = hubbard_from_scf(scf.RHF, N, N, U, 'y')
    rhf.damp = -4.0
    rhf.scf()

    rhf_results = tuple([rhf.e_tot, rhf.mo_coeff, rhf.mo_energy])

    with open('calculated/{}-site/amps_and_scf_eq/rhf_results_u_{}.p'.format(N, U), 'wb') as fp:
        pickle.dump(rhf_results, fp)

    # Run reference calculation
    cc_ref = RCCSD_UNIT(rhf)
    _, energy_ref, amps_ref = root_solver(cc_ref, conv_tol=1e-10)

    cc_results = tuple([energy_ref, amps_ref])

    with open('calculated/{}-site/amps_and_scf_eq/cc_results_u_{}.p'.format(N, U), 'wb') as fp:
        pickle.dump(cc_results, fp)

    all_amps = []
    for idx, rank in enumerate(rankst):
        tim = time.process_time()
        cc = RCCSD_nCPD_LS_T_HUB(rhf, rankt={'t2': rank})
        converged, energy, amps = classic_solver(
            cc, lam=1.8, conv_tol_energy=1e-14,
            conv_tol_amps=1e-10, max_cycle=40000,
            verbose=logger.NOTE)
        if not converged:
            Warning(
                'Warning: N = {}, U = {} '
                'Rank = {} did not converge'.format(N, U, rank)
            )
        all_amps.append(tuple([energy, rank, amps]))
        elapsed = time.process_time() - tim
        print('Step {} out of {}, rank = {}, time: {}'.format(
            idx + 1, len(rankst), rank, elapsed
        ))

    with open('calculated/{}-site/amps_and_scf_eq/energy_rank_amps_u_{}.p'.format(N, U), 'wb') as fp:
        pickle.dump(all_amps, fp)


def calc_err_vs_r_cpd():
    """
    Calculates differences between nCPD approximated and full CC
    at weak correlation
    """
    rankst = RANKS_T.copy()

    rhf = hubbard_from_scf(scf.RHF, N, N, U, 'y')
    rhf.max_cycle = 1
    rhf.scf()
    with open('calculated/{}-site/amps_and_scf_eq/rhf_results_u_{}.p'.format(N, U), 'rb') as fp:
        curr_scf = pickle.load(fp)

    e_scf, mo_coeff, mo_energy = curr_scf

    with open('calculated/{}-site/amps_and_scf_eq/cc_results_u_{}.p'.format(N, U), 'rb') as fp:
        energy_ref, _ = pickle.load(fp)

    with open('calculated/{}-site/amps_and_scf_eq/energy_rank_amps_u_{}.p'.format(N, U), 'rb') as fp:
        cc_solutions = pickle.load(fp)

    energies = []
    deltas = []
    for idx, cc_solution in enumerate(cc_solutions):
        tim = time.process_time()

        energy, rank, amps = cc_solution

        energies.append(energy)
        deltas.append(energy - energy_ref)
        elapsed = time.process_time() - tim
        print('Step {} out of {}, rank = {}, time: {}'.format(
            idx + 1, len(rankst), rank, elapsed
        ))

    results = np.column_stack((rankst, energies, deltas))
    np.savetxt(
        'calculated/{}-site/err_vs_rank_u_{}.txt'.format(N, U),
        results, header='Rank Energy Delta', fmt=('%i', '%e', '%e')
    )
    rankst, energies, deltas = np.loadtxt(
        'calculated/{}-site/err_vs_rank_u_{}.txt'.format(N, U), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    plt.plot(np.log(rankst) / np.log(N), np.log10(np.abs(deltas)))
    plt.xlabel('log$_N(R)$')
    plt.ylabel('log($\Delta$)')
    plt.title('Error dependence on rank in weak correlation regime')
    plt.show()


def plot_err_vs_r_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set data to use
    ns = [6, 10, 14, 18]  # , 22]

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(ns)))
    fig, ax = plt.subplots()

    for n, color in zip(ns, colors):
        rankst, _, deltas = np.loadtxt(
            'calculated/{}-site/err_vs_rank_u_{}.txt'.format(n, U), unpack=True)
        ax.plot(np.log(rankst) / np.log(n), np.log10(np.abs(deltas)),
                color=color, marker=None)

    plt.xlabel('log$_N(R)$')
    plt.ylabel('log($\Delta$)')
    plt.ylim(-9, 0)
    plt.title(
        'Error dependence on rank in weak correlation regime, U = {}'.format(U))
    plt.legend(['{} sites'.format(n) for n in ns])
    fig.show()

    fig.savefig('figures/err_vs_r_u_{}_cpd.eps'.format(U))


if __name__ == '__main__':
    calc_solutions_diff_r_eq_cpd()
    calc_err_vs_r_cpd()
