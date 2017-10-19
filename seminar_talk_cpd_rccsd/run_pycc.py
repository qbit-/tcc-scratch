from pyscf import gto, scf, cc
mol = gto.M(atom='H 0 0 0; H 0 0 1')
mf = scf.RHF(mol).run()
cc.CCSD(mf).run()

from pyscf import gto, scf, cc
mol = gto.M(atom='H 0 0 0; H 0 0 1')
mf = scf.RHF(mol).run()
cc.RCCSD(mf).run()
