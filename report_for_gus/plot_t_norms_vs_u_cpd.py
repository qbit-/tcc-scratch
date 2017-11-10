import numpy as np
import time
import pickle
from pyscf.lib import logger
from pyscf import scf
from tcc.hubbard import hubbard_from_scf
from tcc.cc_solvers import (classic_solver, gradient_solver,
                            root_solver, residual_diis_solver)
from tcc.rccsd import RCCSD, RCCSD_UNIT
from tcc.rccsd_cpd import RCCSD_nCPD_LS_T_HUB
from tcc.cpd import cpd_normalize

# Set up parameters of the script
N = 10
U_VALUES = np.linspace(1, 10, num=20)
# RANKS_T = np.array([5, 7, 8, 10, 12, 20, 25, 40]).astype('int')
RANKS_T = np.array([11, 13, 15, 17, 19]).astype('int')
l1 = [1, ] * 4 + [3, ] * 13 + [5, ] * 3
l2 = [1, ] * 2 + [2, ] * 2 + [3, ] * 10 + [5, ] * 6
l3 = [1, ] * 2 + [2, ] * 2 + [3, ] * 5 + [5, ] * 5 + [7, ] * 6
l4 = [1, ] * 2 + [3, ] * 2 + [8, ] * 5 + [10, ] * 4 + [14, ] * 4 + [18, ] * 3
#LAMBDAS = [l1, l2, l3, l3, l3, l4, l4, l4]
LAMBDAS = [l4, l4, l4, l4, l4]


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


def calc_solutions_diff_u_cpd():
    """
    Run RCCSD-CPD for all corellation strengths
    """
    # Set up parameters of the script
    us = U_VALUES.copy()
    lambdas = LAMBDAS.copy()
    rankst = RANKS_T.copy()

    results = np.array(us)
    results_t1 = np.array(us)
    results_t2 = np.array(us)

    # Run all scfs here so we will have same starting points for CC
    run_scfs(N, us, 'calculated/{}-site/scfs_different_u_t1.p'.format(N))
    with open(
            'calculated/{}-site/scfs_different_u_t1.p'.format(N), 'rb'
    ) as fp:
        ref_scfs = pickle.load(fp)

    for idxr, rank in enumerate(rankst):
        t1_norms = []
        energies = []
        t2_norms = []
        # timb = time.process_time()
        solutions = []
        for idxu, (u, curr_scf) in enumerate(zip(us, ref_scfs)):
            tim = time.process_time()
            rhf = hubbard_from_scf(scf.RHF, N, N, u, 'y')
            rhf.max_cycle = 1
            rhf.scf()
            e_scf, mo_coeff, mo_energy = curr_scf
            cc = RCCSD_nCPD_LS_T_HUB(rhf, rankt={'t2': rank},
                                     mo_coeff=mo_coeff,
                                     mo_energy=mo_energy)
            converged = False
            if idxu == 0:
                converged, energy, amps = classic_solver(
                    cc, lam=lambdas[idxr][idxu], conv_tol_energy=1e-8,
                    conv_tol_amps=1e-7, max_cycle=50000,
                    verbose=logger.NOTE)
            else:
                converged, energy, amps = classic_solver(
                    cc, lam=lambdas[idxr][idxu], conv_tol_energy=1e-8,
                    conv_tol_amps=1e-8, max_cycle=60000,
                    verbose=logger.NOTE, amps=amps)
            solutions.append((u, curr_scf, amps))
            if np.isnan(energy):
                Warning(
                    'Warning: N = {}, U = {} '
                    'Rank = {} did not converge'.format(N, u, rank)
                )

            energies.append(energy + e_scf)
            norms = amps.map(np.linalg.norm)
            if not np.isnan(energy):
                t1_norms.append(norms.t1)
            else:
                t1_norms.append(np.nan)

            if not np.isnan(energy):
                t2_norms.append(amps.t2.xlam[0, 0])
            else:
                t2_norms.append(np.nan)

            elapsed = time.process_time() - tim
            print('Step {} out of {}, rank = {}, time: {}\n'.format(
                idxu + 1, len(us), rank, elapsed
            ))

        results = np.column_stack((results, energies))
        results_t1 = np.column_stack((results_t1, t1_norms))
        results_t2 = np.column_stack((results_t2, t2_norms))
        with open('amps_and_scf_rank_{}.p'.format(rank), 'wb') as fp:
            pickle.dump(solutions, fp)


def calc_t1_norm_vs_u_cpd():
    """
    Collect t1 from calculated solutions into a table
    """
    # Set up parameters of the script
    rankst = RANKS_T.copy()

    with open('calculated/{}-site/amps_and_scf/amps_and_scf_rank_{}.p'.format(N, rankst[0]), 'rb') as fp:
        solutions = pickle.load(fp)
    us = [sol[0] for sol in solutions]
    results_t1 = us

    for idxr, rank in enumerate(rankst):
        with open('calculated/{}-site/amps_and_scf/amps_and_scf_rank_{}.p'.format(N, rank), 'rb') as fp:
            solutions = pickle.load(fp)
        t1_norms = []
        for idxu, (u, curr_scf, amps) in enumerate(solutions):
            tim = time.process_time()
            t1_norm = amps.map(np.linalg.norm).t1
            t1_norms.append(t1_norm)

            elapsed = time.process_time() - tim
            print('Step {} out of {}, rank = {}, time: {}\n'.format(
                idxu + 1, len(solutions), rank, elapsed
            ))

        results_t1 = np.column_stack((results_t1, t1_norms))

    np.savetxt(
        'calculated/{}-site/t1_norm_vs_u.txt'.format(N),
        results_t1,
        header='U ' + ' '.join('R={}'.format(rr) for rr in rankst)
    )

    us, *t1_l = np.loadtxt(
        'calculated/{}-site/t1_norm_vs_u.txt'.format(N), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(us, t1_l[0])
    plt.xlabel('$U / t$')
    plt.ylabel('$||T^{1}||$')
    plt.title('$T^{1}$ norm behavior for different ranks')
    fig.show()


def plot_t1_vs_u_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    rankst = RANKS_T.copy()

    us, *t1_norms_l = np.loadtxt(
        'calculated/{}-site/t1_norm_vs_u.txt'.format(N), unpack=True)

    if len(t1_norms_l) != len(rankst):
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(rankst) + 1))
    fig, ax = plt.subplots()

    rankst = rankst[[0, 1, 2, 3, 5, 6]]  # Cut out some elements
    t1_norms_l = t1_norms_l[:4] + t1_norms_l[5:]
    for idxr, (rank, color) in enumerate(zip(rankst, colors)):
        if idxr == 8:
            marker = '^'
            ls = ''
        else:
            marker = None
            ls = '-'

        ax.plot(us, t1_norms_l[idxr],
                color=color, ls=ls, marker=marker)

    us_rccsd, t1_norms_rccsd_l = np.loadtxt(
        'calculated/{}-site/t1_norm_vs_u_rccsd.txt'.format(N), unpack=True)
    ax.plot(us_rccsd, t1_norms_rccsd_l,
            color='k', ls='', marker='^')

    plt.xlabel('$U / t$')
    plt.ylabel('$||T^{1}||$')
    plt.title(
        'Spatial symmetry breaking for different ranks, {} sites'.format(N)
    )
    # plt.ylim(None, None)
    # plt.xlim(1, 4)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst]
        + ['RCCSD', ],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/t1_norms_vs_u_{}_sites.eps'.format(N))


def calc_t2_norm_vs_u_cpd():
    """
    Collect t2 from calculated solutions into a table
    """
    # Set up parameters of the script
    from tcc.cpd import ncpd_rebuild
    rankst = RANKS_T.copy()

    with open('calculated/{}-site/amps_and_scf/amps_and_scf_rank_{}.p'.format(N, rankst[0]), 'rb') as fp:
        solutions = pickle.load(fp)
    us = [sol[0] for sol in solutions]
    results_t2 = us

    t2names = ['xlam', 'x1', 'x2', 'x3', 'x4']
    for idxr, rank in enumerate(rankst):
        with open('calculated/{}-site/amps_and_scf/amps_and_scf_rank_{}.p'.format(N, rank), 'rb') as fp:
            solutions = pickle.load(fp)
        t2_norms = []
        for idxu, (u, curr_scf, amps) in enumerate(solutions):
            tim = time.process_time()
            t2f = ncpd_rebuild([amps.t2[name] for name in t2names])
            t2_norm = np.linalg.norm(t2f)
            t2_norms.append(t2_norm)

            elapsed = time.process_time() - tim
            print('Step {} out of {}, rank = {}, time: {}\n'.format(
                idxu + 1, len(solutions), rank, elapsed
            ))

        results_t2 = np.column_stack((results_t2, t2_norms))

    np.savetxt(
        'calculated/{}-site/t2_norm_vs_u.txt'.format(N),
        results_t2,
        header='U ' + ' '.join('R={}'.format(rr) for rr in rankst)
    )

    us, *t2_l = np.loadtxt(
        'calculated/{}-site/t2_norm_vs_u.txt'.format(N), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(us, t2_l[0])
    plt.xlabel('$U / t$')
    plt.ylabel('$||T^{1}||$')
    plt.title('$T^{1}$ norm behavior for different ranks')
    fig.show()


def calc_t2_norm_vs_u_rccsd():
    """
    Calculate norms of T2 in conventional RCCSD
    """
    us = U_VALUES.copy()
    lambdas = LAMBDAS.copy()

    with open(
            'calculated/{}-site/scfs_different_u_t1.p'.format(N), 'rb'
    ) as fp:
        ref_scfs = pickle.load(fp)

    t1_norms = []
    energies = []
    t2_norms = []
    # timb = time.process_time()
    solutions = []
    amps = None
    for idxu, (u, curr_scf) in enumerate(zip(us, ref_scfs)):
        rhf = hubbard_from_scf(scf.RHF, N, N, u, 'y')
        rhf.max_cycle = 1
        rhf.scf()
        e_scf, mo_coeff, mo_energy = curr_scf
        cc = RCCSD_UNIT(rhf,
                        mo_coeff=mo_coeff,
                        mo_energy=mo_energy)
        converged, energy, amps = residual_diis_solver(
            cc, conv_tol_energy=1e-9,
            conv_tol_res=1e-9,
            max_cycle=10000,
            verbose=logger.NOTE, amps=None, lam=14)
        norms = amps.map(np.linalg.norm)
        t1_norms.append(norms.t1)
        t2_norms.append(norms.t2)
        energies.append(energy)
        solutions.append([u, curr_scf, amps])
        print('Step {} out of {}'.format(idxu + 1, len(us)))

    t1_norms = np.column_stack((us, t1_norms))
    t2_norms = np.column_stack((us, t2_norms))

    np.savetxt(
        'calculated/{}-site/t1_norm_vs_u_rccsd.txt'.format(N),
        t1_norms,
        header='U |t1|'
    )

    np.savetxt(
        'calculated/{}-site/t2_norm_vs_u_rccsd.txt'.format(N),
        t2_norms,
        header='U |t2|'
    )
    with open(
            'calculated/{}-site/amps_and_scf_rccsd.p'.format(N),
            'wb') as fp:
        pickle.dump(solutions, fp)

    us, *t2_l = np.loadtxt(
        'calculated/{}-site/t2_norm_vs_u_rccsd.txt'.format(N), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(us, t2_l[0])
    plt.xlabel('$U / t$')
    plt.ylabel('$||T^{2}||$')
    plt.title('$T^{2}$ norm behavior for different ranks')
    fig.show()


def plot_t2_vs_u_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    rankst = RANKS_T.copy()

    us, *t2_norms_l = np.loadtxt(
        'calculated/{}-site/t2_norm_vs_u.txt'.format(N), unpack=True)

    if len(t2_norms_l) != len(rankst):
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(rankst) + 1))
    fig, ax = plt.subplots()

    rankst_sel = rankst[[0, 1, 2, 3, 5, 6]]  # Cut out some elements
    t2_norms_l_sel = t2_norms_l[:4] + t2_norms_l[5:7]
    for idxr, (rank, color) in enumerate(zip(rankst_sel, colors)):
        if idxr == 6:
            marker = '^'
            ls = ''
        else:
            marker = None
            ls = '-'
        ax.plot(us, t2_norms_l_sel[idxr],
                color=color, ls=ls, marker=marker)

    us_rccsd, t2_norms_rccsd_l = np.loadtxt(
        'calculated/{}-site/t2_norm_vs_u_rccsd.txt'.format(N), unpack=True)
    ax.plot(us_rccsd, t2_norms_rccsd_l,
            color='k', ls='', marker='^')

    plt.xlabel('$U / t$')
    plt.ylabel('$||T^{2}||$')
    plt.title(
        '$T^{{2}}$ amplitude norm for different ranks, {0} sites'.format(N)
    )
    # plt.ylim(None, None)
    # plt.xlim(1, 4)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst_sel] +
        ['RCCSD', ],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/t2_norms_vs_u_{}_sites.eps'.format(N))


if __name__ == '__main__':
    calc_solutions_diff_u_cpd()
