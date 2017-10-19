import numpy as np
import time
import pickle
from pyscf.lib import logger
from pyscf import scf
from tcc.hubbard import hubbard_from_scf
from tcc.cc_solvers import (classic_solver, root_solver)
from tcc.rccsd import RCCSD_UNIT
from tcc.rccsd_cpd import RCCSD_CPD_LS_T_HUB
from tcc.cpd import cpd_normalize


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


def calc_t1_norm_vs_u_cpd():
    """
    Plot T1 norm of RCCSD-CPD for all corellation strengths
    """
    # Set up parameters of the script
    N = 10
    us = np.linspace(1, 10, num=10)
    lambdas = [3, ] * 6 + [4, ] * 3 + [4, ]
    rankst = np.array([5, 7, 8, 10, 12, 20]).astype('int')

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
        for idxu, (u, curr_scf) in enumerate(zip(us, ref_scfs)):
            tim = time.process_time()
            rhf = hubbard_from_scf(scf.RHF, N, N, u, 'y')
            rhf.max_cycle = 1
            rhf.scf()
            e_scf, mo_coeff, mo_energy = curr_scf
            cc = RCCSD_CPD_LS_T_HUB(rhf, rankt=rank,
                                    mo_coeff=mo_coeff,
                                    mo_energy=mo_energy)
            converged = False
            if idxu == 0:
                converged, energy, amps = classic_solver(
                    cc, lam=lambdas[idxu], conv_tol_energy=1e-8,
                    conv_tol_amps=1e-7, max_cycle=20000,
                    verbose=logger.NOTE)
            else:
                converged, energy, amps = classic_solver(
                    cc, lam=lambdas[idxu], conv_tol_energy=1e-8,
                    conv_tol_amps=1e-7, max_cycle=30000,
                    verbose=logger.NOTE)
            if np.isnan(energy):
                Warning(
                    'Warning: N = {}, U = {} '
                    'Rank = {} did not converge'.format(N, u, rank)
                )

            energies.append(energy + e_scf)
            if not np.isnan(energy):
                t1_norms.append(np.linalg.norm(amps.t1))
            else:
                t1_norms.append(np.nan)

            if not np.isnan(energy):
                tmp, _ = cpd_normalize(amps[1:])
                t2_norms.append(tmp[0])
            else:
                t2_norms.append(np.nan)

            elapsed = time.process_time() - tim
            print('Step {} out of {}, rank = {}, time: {}\n'.format(
                idxu + 1, len(us), rank, elapsed
            ))

        results = np.column_stack((results, energies))
        results_t1 = np.column_stack((results_t1, t1_norms))
        results_t2 = np.column_stack((results_t2, t2_norms))
        # elapsedb = time.process_time() - timb
        # print('Batch {} out of {}, rank = {}, time: {}'.format(
        #  0 + 1, len(rankst), rank, elapsedb))

    # np.savetxt(
    #     'calculated/{}-site/t1_norm_vs_u_energies.txt'.format(N),
    #     results,
    #     header='U '+' '.join('R={}'.format(rr) for rr in rankst)
    # )
    # np.savetxt(
    #     'calculated/{}-site/t1_norm_vs_u.txt'.format(N),
    #     results_t1,
    #     header='U '+' '.join('R={}'.format(rr) for rr in rankst)
    # )
    np.savetxt(
        'calculated/{}-site/lam_1_vs_u.txt'.format(N),
        results_t2,
        header='U '+' '.join('R={}'.format(rr) for rr in rankst)
    )

    us, *t1_norms_l = np.loadtxt(
        'calculated/{}-site/t1_norm_vs_u.txt'.format(N), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(us, t1_norms)
    plt.xlabel('$U$')
    plt.ylabel('$||T^{1}||$')
    plt.title('Energy behavior for different ranks')
    fig.show()


def plot_t1_vs_u_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    N = 10
    rankst = np.array([5, 7, 8, 10, 12, 20]).astype('int')

    us, *t1_norms_l = np.loadtxt(
        'calculated/{}-site/t1_norm_vs_u.txt'.format(N), unpack=True)

    if len(t1_norms_l) != len(rankst):
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(rankst)+1))
    fig, ax = plt.subplots()

    rankst = rankst[:5]  # Cut out some elements
    for idxr, (rank, color) in enumerate(zip(rankst, colors)):
        ax.plot(us, t1_norms_l[idxr],
                color=color, marker=None)
    plt.xlabel('$U / t$')
    plt.ylabel('$||T^{1}||$')
    plt.title(
        'Spatial symmetry breaking for different ranks, {} sites'.format(N)
    )
    # plt.ylim(None, None)
    # plt.xlim(1, 4)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/t1_norms_vs_u_{}_sites.eps'.format(N))


if __name__ == '__main__':
    calc_t1_norm_vs_u_cpd()
