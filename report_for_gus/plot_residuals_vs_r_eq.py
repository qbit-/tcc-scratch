"""
This script plots residuals of singles and doubles
in the weak correlation regime versus the rank of nCPD
decomposition for Hubbard model
"""
import pickle
import numpy as np
import time
from pyscf.lib import logger
from pyscf import scf
from tcc.hubbard import hubbard_from_scf
from tcc.cc_solvers import (classic_solver, root_solver)
from tcc.rccsd import RCCSD_UNIT
from tcc.rccsd_cpd import RCCSD_nCPD_LS_T_HUB
from tcc.cpd import ncpd_rebuild

# Set up parameters of the script
N = 10
U = 2
RANKS_T = np.round(N**np.linspace(0.2, 1.8, num=14)).astype('int')


def calc_residuals_vs_r_cpd_eq():
    """
    Calculates t1 residuals in nCPD approximated and full CC
    at weak correlation
    """
    rankst = RANKS_T.copy()

    rhf = hubbard_from_scf(scf.RHF, N, N, U, 'y')
    rhf.max_cycle = 1
    rhf.scf()
    with open('calculated/{}-site/amps_and_scf_eq/rhf_results_u_{}.p'.format(N, U), 'rb') as fp:
        curr_scf = pickle.load(fp)

    e_scf, mo_coeff, mo_energy = curr_scf

    with open('calculated/{}-site/amps_and_scf_eq/cc_results_u_{}.p'.format(N, U), 'wb') as fp:
        energy_ref, _ = pickle.load(fp)

    with open('calculated/{}-site/amps_and_scf_eq/energy_rank_amps_u_{}.p'.format(N, U), 'rb') as fp:
        cc_solutions = pickle.load(fp)

    r1_results = []
    r2_results = []
    t2names = ['xlam', 'x1', 'x2', 'x3', 'x4']
    for idx, cc_solution in enumerate(cc_solutions):
        tim = time.process_time()

        energy, rank, amps = cc_solution

        cc = RCCSD_nCPD_LS_T_HUB(rhf, rankt={'t2': rank})
        h = cc.create_ham()
        res_norms = cc.calc_residuals(h, amps).map(np.linalg.norm)
        r1_results.append(res_norms.t1)
        r2_results.append(res_norms.t2)

        elapsed = time.process_time() - tim
        print('Step {} out of {}, rank = {}, time: {}'.format(
            idx + 1, len(rankst), rank, elapsed
        ))

    r1_results = np.column_stack((rankst, r1_results))
    r2_results = np.column_stack((rankst, r2_results))
    np.savetxt(
        'calculated/{}-site/r1_vs_rank_u_{}.txt'.format(N, U),
        r1_results, header='Rank |r1|', fmt=('%i', '%e',)
    )
    np.savetxt(
        'calculated/{}-site/r2_vs_rank_u_{}.txt'.format(N, U),
        r2_results, header='Rank |r2|', fmt=('%i', '%e',)
    )
    rankst, energies, deltas = np.loadtxt(
        'calculated/{}-site/r1_vs_rank_u_{}.txt'.format(N, U), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    plt.plot(np.log(rankst) / np.log(N), np.log10(np.abs(deltas)))
    plt.xlabel('log$_N(R)$')
    plt.ylabel('log($|r1|$)')
    plt.title('Singles residual dependence on rank in weak correlation regime')
    plt.show()


def plot_r1_vs_r_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set data to use
    ns = [6, 10, 14, 18, 22]

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(ns)))
    fig, ax = plt.subplots()

    for n, color in zip(ns, colors):
        rankst, _, deltas = np.loadtxt(
            'calculated/{}-site/r1_vs_rank_u_{}.txt'.format(n), unpack=True)
        ax.plot(np.log(rankst) / np.log(n), np.log10(np.abs(deltas)),
                color=color, marker=None)

    plt.ylim(-10, 0)
    plt.xlabel('log$_N(R)$')
    plt.ylabel('log($|r1|$)')
    plt.title('Singles residual dependence on rank in weak correlation regime')
    plt.legend(['{} sites'.format(n) for n in ns])
    fig.show()

    fig.savefig('figures/r1_vs_r_u_{}_cpd.eps'.format(U))


def plot_r2_vs_r_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set data to use
    ns = [6, 10, 14, 18, 22]

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(ns)))
    fig, ax = plt.subplots()

    for n, color in zip(ns, colors):
        rankst, _, deltas = np.loadtxt(
            'calculated/{}-site/r2_vs_rank_u_{}.txt'.format(n), unpack=True)
        ax.plot(np.log(rankst) / np.log(n), np.log10(np.abs(deltas)),
                color=color, marker=None)

    plt.ylim(-10, 0)
    plt.xlabel('log$_N(R)$')
    plt.ylabel('log($|r2|$)')
    plt.title('Singles residual dependence on rank in weak correlation regime')
    plt.legend(['{} sites'.format(n) for n in ns])
    fig.show()

    fig.savefig('figures/r2_vs_r_u_{}_cpd.eps'.format(U))
