#!/usr/bin/python3.6
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
    from tcc.rccsd_mul import RCCSD_MUL_RI
    from tcc.cc_solvers import residual_diis_solver

    tim = time.process_time()

    if (not isfile(workdir + 'ccsd_results.p')
            or not isfile(workdir + 'RCCSD.txt')):

        cc = RCCSD_MUL_RI(rhf,
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

    nbasis = mol.nao_nr()

    rhf = scf.density_fit(scf.RHF(mol))
    rhf.max_cycle = 1
    rhf.scf()

    # Get CCSD results to generate initial guess

    with open(workdir + 'ccsd_results.p', 'wb') as fp:
        energy_ref, amps_ref = pickle.load(fp)

    # Run RCCSD_RI_CPD
    from tcc.rccsd_cpd import RCCSD_nCPD_LS_T
    from tcc.cc_solvers import classic_solver
    from tcc.tensors import Tensors
    from tcc.cpd import ncpd_initialize, als_dense

    ranks_t = [el * nbasis for el in RANKS_T_FACTOR]

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
                ncpd_initialize(amps_ref.shape, rank),
                amps_ref, max_cycle=100, tensor_format='ncpd'
            )
            t2names = ['xlam', 'x1', 'x2', 'x3', 'x4']
            amps_guess = Tensors(t1=amps_ref.t1,
                                 t2=Tensors(zip(t2names, t2x)))

            converged, energy, amps = classic_solver(
                cc, conv_tol_energy=1e-9, conv_tol_res=1e-8,
                max_cycle=500,
                verbose=logger.INFO,
                amps=amps_guess)

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
        ranks_t, energies, deltas, header='Rank Energy Delta',
        fmt=('%i', '%e', '%e',)
    )


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
            results.append((subdir, energies, deltas))
        else:
            results.append((subdir,
                            [np.nan, ] * row_len, [np.nan, ] * row_len))
    with open('RESULTS.txt', 'w') as fp:
        fp.write('System & '
                 + ' & '.join('{:.1f}*N'.format(rr) for rr in RANKS_T_FACTOR)
                 + '\n')
        for res in results:
            fp.write(
                '{} & '.format(res[0])
                + ' & '.join('{:.3f}'.format(ee * 1000) for ee in res[2])
                + '\n')


if __name__ == '__main__':
    if (len(sys.argv) != 2 or not isdir(sys.argv[1])):
        raise ValueError('usage: run_system.py folder')
    wd = sys.argv[1]
    wd = wd.rstrip('/') + '/'  # Make sure we have the workdir
    # with the trailing slash

    # Run RHF
    build_rhf(wd)

    # Run RCCSD
    run_cc_ref(wd)
