from pyscf import gto, scf, rt, ehrenfest
from pyscf import lib, lo

mol =   gto.Mole( atom='''
H    0.0000000    0.0000000    0.3540000
H    0.0000000    0.0000000   -0.3540000
'''
, basis='cc-pvdz', symmetry=False).build()

mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()

dm = mf.make_rdm1()
fock = mf.get_fock()
ehmd = MDHF(mf)
ehmd.verbose = 0
ehmd.maxstep_N = 10
ehmd.maxstep_Ne = 5
ehmd.maxstep_e = 10
ehmd.prop_method = "mmut"
ehmd.dt_N      = 10
ehmd.dt_Ne     = 2
ehmd.dt_e      = 0.2
ehmd.kernel(dm_ao_init=dm)