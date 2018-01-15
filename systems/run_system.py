#!/home/ilias/.local/bin/python3.6
"""
These scripts are for running molecular systems
with RCCSD-CPD
"""
import numpy as np
import pickle
import time
import sys
import os
from os.path import isfile, isdir

from pyscf import gto
from pyscf import scf

# Set up global parameters of the script
BASIS = 'cc-pvdz'
RANKS_T_FACTOR = [1, 1.5, 2]


def load_geom(workdir):
    """
    Loads geometry from geom.dat in the specified directory
    """
    atoms = []
    with open(workdir + 'geom.xyz', 'r') as fp:
        fp.readline()  # Skip first two lines - ugly but works
        fp.readline()
        for line in fp:
            elem, *coords = line.split()
            atoms.append([elem, tuple(coords)])

    return atoms


def build_rhf(workdir):
    print('Running SCF')
    if (not isfile(workdir + 'rhf_results.p')
            or not isfile(workdir + 'rhf_results.p')):

        mol = gto.Mole()
        mol.atom = load_geom(workdir)
        mol.basis = BASIS
        mol.build()
        rhf = scf.RHF(mol)
        rhf = scf.density_fit(scf.RHF(mol))
        rhf.scf()

        rhf_results = tuple([rhf.e_tot, rhf.mo_coeff, rhf.mo_energy, mol.atom])

        with open(workdir + 'rhf_results.p', 'wb') as fp:
            pickle.dump(rhf_results, fp)

        np.savetxt(
            workdir + 'RHF.txt',
            np.array([rhf.e_tot, ]),
            header='Energy'
        )


def run_cc_ref(workdir):
    from pyscf.lib import logger

    print('Running CC')

    # Restore RHF object
    with open(workdir + 'rhf_results.p', 'rb') as fp:
        rhf_results = pickle.load(fp)

    e_tot, mo_coeff, mo_energy, atom = rhf_results
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = BASIS
    mol.build()

    rhf = scf.density_fit(scf.RHF(mol))
    rhf.max_cycle = 1
    rhf.scf()

    # Run RCCSD_RI
    from tcc.rccsd import RCCSD_DIR_RI
    from tcc.cc_solvers import residual_diis_solver

    tim = time.process_time()

    if (not isfile(workdir + 'ccsd_results.p')
            or not isfile(workdir + 'RCCSD.txt')):

        cc = RCCSD_DIR_RI(rhf,
                          mo_coeff=mo_coeff,
                          mo_energy=mo_energy)

        converged, energy, amps = residual_diis_solver(
            cc, conv_tol_energy=1e-9, conv_tol_res=1e-8,
            max_cycle=500,
            verbose=logger.INFO)

        ccsd_results = [energy, amps]
        with open(workdir + 'ccsd_results.p', 'wb') as fp:
            pickle.dump(ccsd_results, fp)

        np.savetxt(
            workdir + 'RCCSD.txt',
            np.array([energy, ]),
            header='Energy'
        )

    elapsed = time.process_time() - tim
    print('Done reference RCCSD, time: {}'.format(elapsed))


def run_cc_cpd(workdir):
    from pyscf.lib import logger

    print('Running CC-CPD')

    # Restore RHF object
    with open(workdir + 'rhf_results.p', 'rb') as fp:
        rhf_results = pickle.load(fp)

    e_tot, mo_coeff, mo_energy, atom = rhf_results
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = BASIS
    mol.build()

    # nbasis = mol.nao_nr()

    rhf = scf.density_fit(scf.RHF(mol))
    rhf.max_cycle = 1
    rhf.scf()

    nbasis_ri = rhf.with_df.get_naoaux()
    # Get CCSD results to generate initial guess

    with open(workdir + 'ccsd_results.p', 'rb') as fp:
        energy_ref, amps_ref = pickle.load(fp)

    # Run RCCSD_RI_CPD
    from tcc.rccsd_cpd import RCCSD_nCPD_LS_T
    from tcc.cc_solvers import classic_solver, update_diis_solver
    from tcc.tensors import Tensors
    from tcc.cpd import ncpd_initialize, als_dense

    ranks_t = [int(el * nbasis_ri) for el in RANKS_T_FACTOR]

    energies = []
    deltas = []

    for idxr, rank in enumerate(ranks_t):
        tim = time.process_time()
        if not isfile(
                workdir +
                'ccsd_results_rank_{}.p'.format(rank)):
            cc = RCCSD_nCPD_LS_T(rhf,
                                 mo_coeff=mo_coeff,
                                 mo_energy=mo_energy)
            # Initial guess
            t2x = als_dense(
                ncpd_initialize(amps_ref.t2.shape, rank),
                amps_ref.t2, max_cycle=100, tensor_format='ncpd'
            )
            t2names = ['xlam', 'x1', 'x2', 'x3', 'x4']
            amps_guess = Tensors(t1=amps_ref.t1,
                                 t2=Tensors(zip(t2names, t2x)))

            converged, energy, amps = update_diis_solver(
                cc, conv_tol_energy=1e-6, conv_tol_res=1e-6,
                max_cycle=500,
                verbose=logger.INFO,
                amps=amps_guess, ndiis=3)

            if not converged:
                energy = np.nan
            ccsd_results = [energy, amps]
            with open(workdir +
                      'ccsd_results_rank_{}.p'.format(rank), 'wb') as fp:
                pickle.dump(ccsd_results, fp)
        else:
            with open(workdir +
                      'ccsd_results_rank_{}.p'.format(rank), 'rb') as fp:
                ccsd_results = pickle.load(fp)
                energy, _ = ccsd_results

        energies.append(energy)
        deltas.append(energy - energy_ref)
        elapsed = time.process_time() - tim

        print('Step {} out of {} done, rank = {}, time: {}'.format(
            idxr + 1, len(ranks_t), rank, elapsed
        ))

    np.savetxt(
        workdir + 'RCCSD_CPD.txt',
        np.column_stack((ranks_t, energies, deltas)),
        header='Rank Energy Delta',
        fmt=('%i', '%e', '%e')
    )
    print('Summary')
    print('=========================================')
    for idxr, rank in enumerate(ranks_t):
        print('{} {}'.format(rank, deltas[idxr]))
    print('=========================================')


def calculate_residuals(workdir):
    print('Running CC-CPD residuals')

    # Restore RHF object
    with open(workdir + 'rhf_results.p', 'rb') as fp:
        rhf_results = pickle.load(fp)

    e_tot, mo_coeff, mo_energy, atom = rhf_results
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = BASIS
    mol.build()

    # nbasis = mol.nao_nr()

    rhf = scf.density_fit(scf.RHF(mol))
    rhf.max_cycle = 1
    rhf.scf()

    nbasis_ri = rhf.with_df.get_naoaux()

    # Load RCCSD_RI_CPD results
    from tcc.rccsd_cpd import RCCSD_nCPD_LS_T
    from tcc.cc_solvers import classic_solver, update_diis_solver
    from tcc.tensors import Tensors
    from tcc.cpd import ncpd_rebuild

    ranks_t = [int(el * nbasis_ri) for el in RANKS_T_FACTOR]

    norms_t1 = []
    norms_t2 = []

    for idxr, rank in enumerate(ranks_t):
        if isfile(
                workdir +
                'ccsd_results_rank_{}.p'.format(rank)):
            with open(workdir +
                      'ccsd_results_rank_{}.p'.format(rank), 'rb') as fp:
                ccsd_results = pickle.load(fp)
                energy, amps = ccsd_results

            # Create cc object
            cc = RCCSD_nCPD_LS_T(rhf,
                                 mo_coeff=mo_coeff,
                                 mo_energy=mo_energy)
            ham = cc.create_ham()
            res = cc.calc_residuals(ham, amps)

            # residuals
            # res_full = Tensors(
            #    t1=res.t1,
            #    t2=ncpd_rebuild([res.t2.xlam, res.t2.x1, res.t2.x2,
            #                     res.t2.x3, res.t2.x4])
            # )

            norms = res.map(np.linalg.norm)

        else:
            raise FileNotFoundError(
                workdir +
                'ccsd_results_rank_{}.p does not exist'.format(rank))

        norms_t1.append(norms.t1)
        norms_t2.append(norms.t2)

    np.savetxt(
        workdir + 'RCCSD_CPD_RES.txt',
        np.column_stack((ranks_t, norms_t1, norms_t2)),
        header='Rank |R1| |R2|',
        fmt=('%i', '%e', '%e')
    )
    print('Summary')
    print('=========================================')
    for idxr, rank in enumerate(ranks_t):
        print('{} {}'.format(rank, norms_t2[idxr]))
    print('=========================================')


def collect_table():
    """
    Build a table with results
    """
    contents = os.listdir()
    dirs = [elem for elem in sorted(contents) if isdir(elem)]
    cwd = os.getcwd()

    row_len = len(RANKS_T_FACTOR)

    results = []

    for subdir in dirs:
        wd = cwd + '/' + subdir + '/'
        if isfile(wd + 'RCCSD_CPD.txt'):
            ranks, energies, deltas = np.loadtxt(
                wd + 'RCCSD_CPD.txt', unpack=True)
        else:
            energies = [np.nan, ] * row_len
            deltas = [np.nan, ] * row_len
        if isfile(wd + 'RCCSD_CPD_RES.txt'):
            ranks, res_t1, res_t2 = np.loadtxt(
                wd + 'RCCSD_CPD_RES.txt', unpack=True)
        else:
            res_t1 = [np.nan, ] * row_len
            res_t2 = [np.nan, ] * row_len

        if isfile(wd + 'RCCSD.txt') and isfile(wd + 'RHF.txt'):
            energy_ref = (
                float(np.loadtxt(
                    wd + 'RCCSD.txt'))
                + float(np.loadtxt(
                    wd + 'RHF.txt'))
                )
        else:
            energy_ref = np.nan
        results.append((subdir, energy_ref, energies, deltas, res_t2))
    with open('RESULTS.txt', 'w') as fp:
        fp.write('System & Energy, H & '
                 + ' & '.join('{:.1f}*N'.format(rr) for rr in RANKS_T_FACTOR)
                 + ' & '
                 + ' & '.join('{:.1f}*N'.format(rr) for rr in RANKS_T_FACTOR)
                 + '\n')
        for res in results:
            fp.write(
                '{} & '.format(res[0])
                + '{:.3f}'.format(res[1])
                + ' & '
                + ' & '.join('{:.3f}'.format(ee * 1000) for ee in res[3])
                + ' & '
                + ' & '.join('{:.3f}'.format(ee) for ee in res[4])
                + '\n')


def run_all():
    """
    Run all systems until we fail!
    """
    contents = os.listdir()
    dirs = [elem for elem in sorted(contents) if isdir(elem)]

    for dirname in dirs:
        print('Working on: {}'.format(dirname))
        run_dir(dirname + '/')
        calculate_residuals(dirname + '/')

    collect_table()


def run_dir(wd):

    # Run RHF
    build_rhf(wd)

    # Run RCCSD
    run_cc_ref(wd)

    # Run RCCSD-CPD
    run_cc_cpd(wd)


if __name__ == '__main__':
    if (len(sys.argv) != 2 or not isdir(sys.argv[1])):
        raise ValueError('usage: run_system.py folder')
    wd = sys.argv[1]
    wd = wd.rstrip('/') + '/'  # Make sure we have the workdir
    # with the trailing slash
    # run_dir(wd)
    # run_all()
    collect_table()
