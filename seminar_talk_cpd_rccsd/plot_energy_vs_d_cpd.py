import numpy as np
import time
import pickle
from pyscf.lib import logger
from pyscf import scf
from tcc.hubbard import hubbard_from_scf
from tcc.cc_solvers import (classic_solver, root_solver)
from tcc.rccsd import RCCSD_UNIT
from tcc.rccsd_cpd import RCCSD_CPD_LS_T


def run_scfs(mols, filename):
    scf_energies = []
    scf_mos = []
    scf_mo_energies = []
    for idm, mol in enumerate(mols):
        rhf = scf.density_fit(scf.RHF(mol))
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


def calc_energy_vs_d_cpd():
    """
    Plot energy of RCCSD-CPD for different distances in dissociation of N2
    """
    # Set up parameters of the script

    basis = 'cc-pvdz'
    dists = np.linspace(0.8, 2.7, num=15)
    lambdas = [3, ] * 6 + [5, ] * 6 + [6, ] * 3
    rankst = np.array([2, 4, 5, 6, 8]).astype('int')

    def make_mol(dist):
        from pyscf import gto
        mol = gto.Mole()
        mol.atom = [
            [7, (0., 0., 0.)],
            [7, (0., 0., dist)]]

        mol.basis = {'N': basis}
        mol.build()
        return mol

    mols = [make_mol(dist) for dist in dists]

    results = np.array(dists)

    # Run all scfs here so we will have same starting points for CC
    run_scfs(mols, 'calculated/{}/scfs_different_dist.p'.format(basis))
    with open('calculated/{}/scfs_different_dist.p'.format(basis), 'rb') as fp:
        ref_scfs = pickle.load(fp)

    for idxr, rank in enumerate(rankst):
        lambdas = [3, ] * 6 + [4, ] * 0 + [6, ] * 6 + [6, ] * 3

        energies = []
        converged = False
        # timb = time.process_time()
        for idxd, (dist, curr_scf) in enumerate(zip(dists, ref_scfs)):
            tim = time.process_time()
            rhf = scf.density_fit(scf.RHF(mols[idxd]))
            rhf.max_cycle = 1
            rhf.scf()
            e_scf, mo_coeff, mo_energy = curr_scf
            cc = RCCSD_CPD_LS_T(rhf, rankt=rank,
                                mo_coeff=mo_coeff,
                                mo_energy=mo_energy)
            if not converged:
                amps = None

            converged, energy, amps = classic_solver(
                cc, lam=lambdas[idxd], conv_tol_energy=1e-8,
                conv_tol_amps=1e-7, max_cycle=30000,
                verbose=logger.NOTE)

            if not converged:
                Warning(
                    'Warning: D = {} Rank = {}'
                    ' did not converge'.format(dist, rank)
                )

            energies.append(energy + e_scf)
            elapsed = time.process_time() - tim
            print('Step {} out of {}, rank = {}, time: {}\n'.format(
                idxd + 1, len(dists), rank, elapsed
            ))

        results = np.column_stack((results, energies))
        # elapsedb = time.process_time() - timb
        # print('Batch {} out of {}, rank = {}, time: {}'.format(
        #  0 + 1, len(rankst), rank, elapsedb))

    np.savetxt(
        'calculated/{}/energy_vs_d.txt'.format(basis),
        results,
        header='Dist '+' '.join('R={}'.format(rr) for rr in rankst)
    )
    us, *energies_l = np.loadtxt(
        'calculated/{}/energy_vs_d.txt'.format(basis), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(dists, energies, marker='o')
    plt.xlabel('$D, \AA$')
    plt.ylabel('$E$, H')
    plt.title('Energy behavior for different ranks')
    fig.show()


def plot_energy_vs_d_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    basis = 'cc-pvdz'
    rankst = np.array([2, 4, 5, 6, 8]).astype('int')

    us, *energies_l = np.loadtxt(
        'calculated/{}/energy_vs_d.txt'.format(basis), unpack=True)

    if len(energies_l) != len(rankst):
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(rankst)))
    fig, ax = plt.subplots()

    rankst = rankst[:4]  # Cut out some elements

    for idxr, (rank, color) in enumerate(zip(rankst, colors)):
        ax.plot(us, energies_l[idxr],
                color=color, marker=None)
    d_ref, e_ref = np.loadtxt(
        'reference/{}/FCI_all_el.txt'.format(basis), unpack=True)
    ax.plot(d_ref, e_ref,
            color='k', marker=None)

    d_cc, e_cc = np.loadtxt(
        'reference/{}/gauRCCSD.txt'.format(basis), unpack=True)
    ax.plot(d_cc, e_cc,
            ':k', marker=None)

    plt.xlabel('$D, \AA$')
    plt.ylabel('$E$, H')
    plt.title('Energy behavior for different ranks, $N_2$ ({})'.format(basis))
    plt.ylim(-107.75, -107.2)
    plt.xlim(0.8, 2.2)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst] + ['Exact', ] + ['RCCSD', ],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/energy_vs_d_{}.eps'.format(basis))

if __name__ == '__main__':
    calc_energy_vs_d_cpd()
