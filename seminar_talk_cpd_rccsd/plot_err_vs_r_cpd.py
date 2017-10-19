import numpy as np
import time
from pyscf.lib import logger

def calc_err_vs_r_cpd():
    """
    Plot error of RCCSD-CPD vs rank for weak corellation
    """
    # Set up parameters of the script
    N = 14
    U = 2

    rankst = np.round(N**np.linspace(0.2, 1.8, num=10)).astype('int')

    # Run RHF calculations
    from pyscf import scf
    from tcc.hubbard import hubbard_from_scf
    rhf = hubbard_from_scf(scf.RHF, N, N, U, 'y')
    rhf.damp = -4.0
    rhf.scf()

    from tcc.cc_solvers import (classic_solver, root_solver)
    from tcc.rccsd import RCCSD_UNIT
    from tcc.rccsd_cpd import RCCSD_CPD_LS_T_HUB
    from tensorly.decomposition import parafac

    # Run reference calculation
    cc_ref = RCCSD_UNIT(rhf)
    _, energy_ref, amps_ref = root_solver(cc_ref, conv_tol=1e-10)

    energies = []
    deltas = []
    for idx, rank in enumerate(rankst):
        tim = time.process_time()
        cc = RCCSD_CPD_LS_T_HUB(rhf, rankt=rank)
        xs = parafac(
            amps_ref.t2, rank, tol=1e-14
        )
        amps_guess = cc.types.AMPLITUDES_TYPE(
            amps_ref.t1, *xs
        )
        converged, energy, amps = classic_solver(
            cc, lam=1.8, conv_tol_energy=1e-14,
            conv_tol_amps=1e-10, max_cycle=10000,
            amps=amps_guess, verbose=logger.NOTE)
        if not converged:
            Warning(
                'Warning: N = {}, U = {} '
                'Rank = {} did not converge'.format(N, U, rank)
            )
        energies.append(energy)
        deltas.append(energy - energy_ref)
        elapsed = time.process_time() - tim
        print('Step {} out of {}, rank = {}, time: {}'.format(
            idx + 1, len(rankst), rank, elapsed
        ))

    results = np.column_stack((rankst, energies, deltas))
    np.savetxt(
        'calculated/{}-site/err_vs_rank.txt'.format(N),
        results, header='Rank Energy Delta', fmt=('%i', '%e', '%e')
    )
    rankst, energies, deltas = np.loadtxt(
        'calculated/{}-site/err_vs_rank.txt'.format(N), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    plt.plot(np.log(rankst)/np.log(N), np.log10(np.abs(deltas)))
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
    ns = [6, 8, 10, 12, 14]

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(ns)))
    fig, ax = plt.subplots()

    for n, color in zip(ns, colors):
        rankst, _, deltas = np.loadtxt(
            'calculated/{}-site/err_vs_rank.txt'.format(n), unpack=True)
        ax.plot(np.log(rankst)/np.log(n), np.log10(np.abs(deltas)),
                color=color, marker=None)

    plt.xlabel('log$_N(R)$')
    plt.ylabel('log($\Delta$)')
    plt.ylim(-10, 0)
    plt.title('Error dependence on rank in weak correlation regime, U = 2')
    plt.legend(['{} sites'.format(n) for n in ns])
    fig.show()

    fig.savefig('figures/err_vs_r_cpd.eps')


if __name__ == '__main__':
    calc_err_vs_r_cpd()
