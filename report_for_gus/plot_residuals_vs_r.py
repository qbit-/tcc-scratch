import numpy as np
import time
import pickle
from pyscf.lib import logger
from pyscf import scf
from tcc.hubbard import hubbard_from_scf
from tcc.cc_solvers import (classic_solver, root_solver)
from tcc.rccsd import RCCSD_UNIT
from tcc.rccsd_cpd import RCCSD_nCPD_LS_T_HUB
from tcc.cpd import cpd_normalize

# Set up parameters of the script
N = 10
RANKS_T = np.array([5, 7, 8, 10, 20, 25, 40]).astype('int')


def calc_resids_norm():
    """
    Plot T1 norm of RCCSD-CPD for all corellation strengths
    """
    # Set up parameters of the script
    rankst = RANKS_T.copy()

    with open('calculated/{}-site/amps_and_scf/amps_and_scf_rank_{}.p'.format(N, rankst[0]), 'rb') as fp:
        solutions = pickle.load(fp)
    us = [sol[0] for sol in solutions]
    results_r1 = us
    results_r2 = us

    for idxr, rank in enumerate(rankst):
        with open('calculated/{}-site/amps_and_scf/amps_and_scf_rank_{}.p'.format(N, rank), 'rb') as fp:
            solutions = pickle.load(fp)
        r1_norms = []
        r2_norms = []
        for idxu, (u, curr_scf, amps) in enumerate(solutions):
            tim = time.process_time()
            rhf = hubbard_from_scf(scf.RHF, N, N, u, 'y')
            rhf.max_cycle = 1
            rhf.scf()
            e_scf, mo_coeff, mo_energy = curr_scf
            cc = RCCSD_nCPD_LS_T_HUB(rhf, rankt={'t2': rank},
                                     mo_coeff=mo_coeff,
                                     mo_energy=mo_energy)
            h = cc.create_ham()
            res = cc.calc_residuals(h, amps)
            norms = res.map(np.linalg.norm)
            r1_norms.append(norms.t1)
            r2_norms.append(norms.t2)

            elapsed = time.process_time() - tim
            print('Step {} out of {}, rank = {}, time: {}\n'.format(
                idxu + 1, len(solutions), rank, elapsed
            ))

        results_r1 = np.column_stack((results_r1, r1_norms))
        results_r2 = np.column_stack((results_r2, r2_norms))

    np.savetxt(
        'calculated/{}-site/r1_norm_vs_u.txt'.format(N),
        results_r1,
        header='U ' + ' '.join('R={}'.format(rr) for rr in rankst)
    )
    np.savetxt(
        'calculated/{}-site/r2_norm_vs_u.txt'.format(N),
        results_r2,
        header='U ' + ' '.join('R={}'.format(rr) for rr in rankst)
    )

    us, *r1_norms_l = np.loadtxt(
        'calculated/{}-site/r1_norm_vs_u.txt'.format(N), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(us, r1_norms_l[0])
    plt.xlabel('$U$')
    plt.ylabel('$||r^{1}||$')
    plt.title('R1 behavior for different ranks')
    fig.show()


def plot_r1_vs_u_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    rankst = RANKS_T.copy()

    us, *r1_norms_l = np.loadtxt(
        'calculated/{}-site/r1_norm_vs_u.txt'.format(N), unpack=True)

    if len(r1_norms_l) != len(rankst):
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(rankst) + 1))
    fig, ax = plt.subplots()

    rankst = rankst  # Cut out some elements
    for idxr, (rank, color) in enumerate(zip(rankst, colors)):
        ax.semilogy(us, r1_norms_l[idxr],
                    color=color, marker=None)
    plt.xlabel('$U / t$')
    plt.ylabel('$||R^{1}||$')
    plt.title(
        'Singles residuals for different ranks, {} sites'.format(N)
    )
    # plt.ylim(None, None)
    # plt.xlim(1, 4)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/r1_norms_vs_u_{}_sites.eps'.format(N))


def plot_r2_vs_u_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    rankst = RANKS_T.copy()

    us, *r2_norms_l = np.loadtxt(
        'calculated/{}-site/r2_norm_vs_u.txt'.format(N), unpack=True)

    if len(r2_norms_l) != len(rankst):
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(rankst) + 1))
    fig, ax = plt.subplots()

    rankst = rankst[:6]  # Cut out some elements
    for idxr, (rank, color) in enumerate(zip(rankst, colors)):
        if idxr == 6:
            ls = ''
            marker = '^'
        else:
            ls = '-'
            marker = None

        ax.plot(us, r2_norms_l[idxr],
                color=color, marker=marker, ls=ls)
    plt.xlabel('$U / t$')
    plt.ylabel('$||R^{2}||$')
    plt.title(
        'Doubles residuals for different ranks, {} sites'.format(N)
    )
    # plt.ylim(None, None)
    # plt.xlim(1, 4)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/r2_norms_vs_u_{}_sites.eps'.format(N))


if __name__ == '__main__':
    pass
