# Author Jingheng Deng (deng.jing.heng223@hotmail.com)

import time
import tempfile

from functools import reduce
import numpy as np
import scipy

from pyscf import gto, scf
from pyscf import lib, lo
from pyscf.lib import logger
from pyscf.data import elements
from pyscf.grad import rhf

class NucDynamics(lib.StreamObject):
    def __init__(self, mf):
        # module control
        self._scf            = mf
        self.mol             = mf.mol

        self.verbose         = mf.verbose
        self._scf.verbose    = 0
        self.max_memory      = mf.max_memory
        self.stdout          = mf.stdout
        self.orth_xtuple     = None
        self.grad            = self._scf.Gradients()
        
        # step control
        self.max_step        = None
        self.step_size       = None
        self.save_step       = None   # save results once after every n step(s)
        
        # input control
        self.dm_ao_init      = self._scf.make_rdm1()
        self.coord_init      = None
        self.momenta_init    = None
        self.force_init      = None
        
        # output control
        
        self.ntime           = None
        self.ncoords         = None
        self.nforce          = None
        self.nmomenta        = None
        self.netot           = None
        assert self._scf.converged is True, "SCF not converged, BOMD method need a converged SCF."
    
    def orthogonalize(self, method='lowdin'):
        import scipy
        mf = self._scf
        s1e = mf.get_ovlp()
        s1e_inv = np.linalg.inv(s1e)

        if method == 'lowdin':
            v = scipy.linalg.sqrtm(s1e_inv)
            v_inv = np.linalg.inv(v)
        return v, v_inv
        
    def get_grad_elec(self, dm_ao):
        mf = self._scf
        mol = mf.mol
        grad = mf.Gradients()

        h1e = mf.get_hcore()
        
        s1 = grad.get_ovlp(mol)
        hcore_deriv = grad().hcore_generator(mol)
        
        dm0_r = dm_ao.real
        dm0_i = dm_ao.imag
        dm0 = dm_ao.real + 1j * dm_ao.imag

        vhf_r = grad.get_veff(mol, dm=dm0_r)
        vhf_i = grad.get_veff(mol, dm=dm0_i)
        vhf = vhf_r + 1j * vhf_i

        veff_r = mf.get_veff(dm0_r)
        veff_i = mf.get_veff(dm0_i)
        veff = veff_r + 1j * veff_i

        fock_ao_r = mf.get_fock(dm=dm0_r, h1e=h1e, vhf=veff_r)
        fock_ao_i = mf.get_fock(dm=dm0_i, h1e=h1e, vhf=veff_i)
        fock_ao = fock_ao_r + 1j * fock_ao_i

        V, V_inv = self.orthogonalize(method='lowdin') ### To be modified

        VVTFP = np.einsum('ij,kj->ik', V, V)
        VVTFP = np.einsum('ij,jk,kl->il', VVTFP, fock_ao, dm0)
        VVTFP += VVTFP.T

        atmlst = range(mol.natm)
        aoslices = mol.aoslice_by_atom()

        de = np.zeros((len(atmlst),3), dtype=dm0.dtype)

        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices [ia,2:]
            h1ao = hcore_deriv(ia)    
            de[k] += np.einsum('xij,ij->x', h1ao, dm0)
            de[k] += np.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2  
            de[k] -= np.einsum('ij,xij->x', VVTFP[p0:p1], s1[:,p0:p1])    

            de[k] += grad().extra_force(ia, locals())  

        return de.real
        
    def get_grad_nuc(self):
        grad = self.grad
        
        return grad.grad_nuc(self.mol)
        
    def get_atom_mass(self, mol):
        '''Get atom mass in atomic units'''
        atom_list = [mol.atom_symbol(i) for i in range(mol.natm)]
        mass_list = []
        amu2au = 1.6605402e-24/9.1093837015e-28

        atom_ref = elements.ELEMENTS
        mass_ref = elements.MASSES

        for iatom in atom_list:
            for index, atom in enumerate(atom_ref):
                if iatom == atom:
                    mass_list.append(mass_ref[index])

        return np.array(mass_list)*amu2au
        
    def update_mol(self, new_coords):
        self.mol = self.mol.set_geom_(new_coords * lib.param.BOHR, inplace=True)
        self._scf.reset(self.mol)
        self._scf.kernel()
        
#         print('update.mol:\n',self._scf.mol/.atom_coords() * lib.param.BOHR)
    
    def update_dm_bomd(self, scf_max_cycle):
        mf = self._scf
        mf.max_cycle = scf_max_cycle
        mf.kernel()
        assert self._scf.converged is True, "SCF not converged, BOMD method need a converged SCF."
        dm_ao = mf.make_rdm1()
        
        return dm_ao
    
    def cal_force(self, dm_ao=None):
        if dm_ao is None:
            dm_ao = self._scf.make_rdm1()
            
        new_force = self.get_grad_elec(dm_ao) + self.get_grad_nuc()
#         self._scf.kernel()
#         new_force = self._scf.Gradients().grad()
        
        return -new_force
        
    def cal_velocity(self, momenta, mass):
        '''Calculate velocity from momenta'''
        mass_reci = np.reciprocal(mass).reshape((1,2))
        vel = momenta * mass_reci.T
        
        return vel
    
    def coord_prop_step(self, coord_this, momenta_this, force_this, step_size):
        '''Propagate the coordinates of nucleus from "this" moment to "next" moment with using Verlet algorithm'''
        
        mass = self.get_atom_mass(self.mol)
        momenta_nexthalf = momenta_this + 1/2 * force_this * step_size
        coord_next       = coord_this + self.cal_velocity(momenta_nexthalf, mass) * step_size
        
        return momenta_nexthalf, coord_next
        
    def momenta_half_prop_step(self, momenta_nexthalf, force_next, step_size):
        '''Propagate the momenta of nucleus from "next_half" moment to "next" moment with using Verlet algorithm'''
        
        momenta_next = momenta_nexthalf + 1/2 * force_next * step_size
        
        return momenta_next
    
    def vib_ene2momenta(self):
        mass = np.array([self.nd.get_atom_mass(self.mol)])
        eV2au = 0.036749405469679
        energy = self.vib_ene_init
        energy = energy * eV2au

        # Convert vibration energy (eV) to momenta (Only for 2-atom molecule)
        # Initially momenta is conserved: P_a = -P_b
        
        mass_r = (mass[0][0] + mass[0][1]) / (mass[0][0] * mass[0][1])
        vector = self.mol.atom_coords()[0] - self.mol.atom_coords()[1]
        direction = vector / (np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)) / 2
        momenta_sq = energy / mass_r * 2 
        momenta = np.sqrt(momenta_sq)
        self.momenta_init = -momenta * np.array((direction,-direction))

        
    def momenta2ene(self, momenta):
        hartree2ev = 27.211386246
        mass = np.array([self.get_atom_mass(self.mol)])
        mass_r = (mass[0][0] + mass[0][1]) / (mass[0][0] * mass[0][1])
        ene = np.linalg.norm(momenta[1]-momenta[0])**2 * mass_r / 2
        
        return ene
    
    def kernel(self, scf_max_cycle=None):
        '''Bohn-Oppenheimer molecular dynamics procedure'''
        
        if self.save_step is None:
            self.save_step = 1
        if scf_max_cycle is None:
            scf_max_cycle = 100     # Default calculation is BOMD which requires a converged scf result.
            
        if self.coord_init is None:
            coord_this = self.mol.atom_coords()
        else:
            coord_this = self.coord_init
            
        if self.momenta_init is None:
            momenta_this = np.zeros(list(coord_this.shape), dtype=np.float64)
            self.momenta_init = momenta_this
        else:
            momenta_this = self.momenta_init
            
        if self.force_init is None:
            force_this = self.cal_force(dm_ao=self.dm_ao_init)
        else:
            force_this = self.force_init
        
        time = 0.0
        etot_this = self._scf.e_tot
        
        self.nforces  = np.zeros([self.max_step // self.save_step + 1] + list(coord_this.shape), dtype=np.float64)
        self.nmomenta = np.zeros([self.max_step // self.save_step + 1] + list(coord_this.shape), dtype=np.float64)
        self.ncoords  = np.zeros([self.max_step // self.save_step + 1] + list(coord_this.shape), dtype=np.float64)
        self.ntime    = np.zeros([self.max_step // self.save_step + 1])
        self.netot    = np.zeros([self.max_step // self.save_step + 1])
        self.nE_tot   = np.zeros([self.max_step // self.save_step + 1])


        
        self.nforces[0]  = force_this
        self.nmomenta[0] = momenta_this
        self.ncoords[0]  = coord_this
        self.ntime[0]    = time
        self._scf.energy_tot(dm=self.dm_ao_init, h1e=self._scf.get_hcore(), vhf=self._scf.get_veff(dm=self.dm_ao_init)).real
        self.nE_tot[0]   = self.momenta2ene(self.momenta_init) + self.netot[0]
            
        istep = 1
        
        while istep < self.max_step+1:
            momenta_nexthalf, coord_next = self.coord_prop_step(coord_this, momenta_this, force_this, self.step_size)
            self.update_mol(coord_next)
#             print('mol.coord of ',istep,': \n',self.mol.atom_coords()*lib.param.BOHR)
            
            dm_ao_next                   = self.update_dm_bomd(scf_max_cycle)
            force_next                   = self.cal_force(dm_ao=dm_ao_next)
            momenta_next                 = self.momenta_half_prop_step(momenta_nexthalf, force_next, self.step_size)
            time += self.step_size
            
            # save
            if (istep / self.save_step) == (istep // self.save_step):
                print('saved step: %d'%(istep//self.save_step))  # need to be replaced with "log" function -- 2020/10/22
                self.nforces[istep//self.save_step]  = force_next
                self.nmomenta[istep//self.save_step] = momenta_next
                self.ncoords[istep//self.save_step]  = coord_next
                self.ntime[istep//self.save_step]    = time
                self.netot[istep//self.save_step]    = self._scf.energy_tot(dm=dm_ao_next, h1e=self._scf.get_hcore(), vhf=self._scf.get_veff(dm=dm_ao_next)).real
                self.nE_tot[istep//self.save_step]   = self.momenta2ene(self.nmomenta[istep//self.save_step]) + self.netot[istep//self.save_step]
            
            # update
            coord_this    = coord_next
            force_this    = force_next
            momenta_this  = momenta_next
            istep +=1
            
        # dump_chk to be done -- 2020/10/22
        
        np.array(self.nforces)
        np.array(self.nmomenta)
        np.array(self.ncoords)
        np.array(self.ntime)
        
        
        
class ElecDynamics(lib.StreamObject):
    '''Real-time HF/DFT'''
    def __init__(self,mf):
        # module control
        self._scf            = mf
        self.mol             = mf.mol

        self.verbose         = mf.verbose
        self._scf.verbose    = 0
        self.max_memory      = mf.max_memory
        self.stdout          = mf.stdout
        self.orth_xtuple     = None
        self.prop_method     = None

        # step control
        self.max_step        = None
        self.step_size       = None
        self.save_step       = None   # save results once after every n step(s)

        # input control
        self.dm_ao_init      = self._scf.make_rdm1()
        self.vhf_ao_init     = self._scf.get_veff(dm=self.dm_ao_init)
        self.apply_efield    = False 

        # output control
        self.ntime           = None
        self.nfock_ao        = None
        self.ndm_ao          = None
        self.netot           = None
        
        
    def orth_ao(self, orth_method=None):
        '''Orthogonalize AOs'''
        assert orth_method is not None, 'Please specify orthogonalization method.'
        mf = self._scf
        if orth_method.lower() == 'lowdin': # non-orthogonalized AO basis to orthogonalized AO basis using lowdin method
            import scipy            
            s1e = mf.get_ovlp()
            s1e_inv = np.linalg.inv(s1e)
            x = scipy.linalg.sqrtm(s1e_inv)
                   
        elif orth_method.lower() == 'caonical': # AO basis to MO basis
            assert mf.converged is True, 'SCF must be converged'
            x = mf.mo_coeff
            
        return x.astype(np.complex128)            

    def ao2orth(self, target, tar_type=None):
        '''Orthogonalize density/fock matrix from AOs'''
        assert tar_type is not None, 'Specify the type of target, dm or fock.'
        
        x, x_T, x_inv, x_inv_T = self.orth_xtuple
        if tar_type == 'dm':
            dm_ao = target
            dm_orth = np.einsum('ij,jk,kl->il', x_inv, dm_ao, x_inv_T)
            
            return dm_orth
        
        elif tar_type == 'fock':
            fock_ao = target
            fock_orth = np.einsum('ij,jk,kl->il', x_T, fock_ao, x)
            
            return fock_orth

    def orth2ao(self, target, tar_type=None):
        '''Transfer density/fock matrix from orthogonal basis to AOs'''
        assert tar_type is not None, 'Specify the type of target, dm or fock.'
        
        x, x_T, x_inv, x_inv_T = self.orth_xtuple        
        if tar_type == 'dm':
            dm_orth = target
            dm_ao = np.einsum('ij,jk,kl->il', x, dm_orth, x_T)
            
            return dm_ao
            
        elif tar_type == 'fock':
            fock_orth = target
            fock_ao = np.eimsum('ij,jk,kl->il', x_inv_T, fock_orth, x_inv)
            
            return fock_ao
        
    def efield(time, direct_vector=None, field_strength=None, field_type=None, period=None, freqency=None):
        '''Generate a electric field vector at the given moment'''
        return field_strength * direct_vector      # To be done
        
    def get_hcore(self, time=None): # To be done -- 2020/10/28
        hcore_ao = self._scf.get_hcore().astype(np.complex128)
        if (self.apply_efield is None) or (time is None):
            return  hcore_ao
#         else:                                     # To be done
#             if self.ele_dip_ao is None:
#                 # the interaction between the system and electric field
#                 self.ele_dip_ao = self._scf.mol.intor_symmetric('int1e_r', comp=3)
            
#             hcore_ao_ = hcore_ao + np.einsum('xij,x->ij', self.ele_dip_ao, self.efield(t)).astype(numpy.complex128)
#             return hcore_ao_
        
        
    def get_fock(self, dm, time):
        h1e = self.get_hcore() 
        vhf = self._scf.get_veff(dm=dm)
        return self._scf.get_fock(dm=dm, h1e=h1e, vhf=vhf)
    
    def check_consistency(self, M_a, M_b, tol_=None):
        '''Consistency check is based on Frobenius norm'''
        assert len(M_a) == len(M_b), 'The dimension of matrix A and B does not match.'
        if tol_ is None:
            tol_ = 1e-7
        
        delta = np.subtract(M_a, M_b)
        F_norm = np.linalg.norm(delta)
        n = len(M_a)
        Xi = F_norm/n

        if Xi < tol_:
            return True
        else:
            return False
    
    def elec_prop_step(self, time, step_size, dm_ao=None, fock_ao=None, dm_prime=None, fock_prime=None):
        '''Propagate density matrix from t_N to t_(N+a) where a depends on the given step_size'''
        if dm_ao is None and dm_prime is not None:
            dm_prime = dm_prime
        elif dm_ao is not None and dm_prime is None:
            dm_prime = self.ao2orth(dm_ao, tar_type='dm')
#             print('elec_prop_step:dm_prime_this\n', dm_prime)
        elif dm_ao is not None and dm_prime is not None:
            error = True
            assert error is False,"For dm, only input dm_ao or dm_prime."
        elif dm_ao is None and dm_prime is None:
            error = True
            assert error is False,"Please input dm_ao or dm_prime."
            
        if fock_ao is None and fock_prime is not None:
            fock_prime = fock_prime
        elif fock_ao is not None and fock_prime is None:
            fock_prime = self.ao2orth(fock_ao, tar_type='fock')
        elif fock_ao is not None and fock_prime is not None:
            error = True
            assert error is False,"For fock, only input fock_ao or fock_prime."
        elif fock_ao is None and fock_prime is None:
            dm_ao = self.orth2ao(dm_prime, tar_type='dm')
            fock_ao = self.get_fock(dm_ao, time)
            fock_prime = self.ao2orth(fock_ao, tar_type='fock')
        
        import scipy
        
        U = scipy.linalg.expm(-1j * step_size * fock_prime)
        dm_prime_next = np.einsum('ij,jk,kl->il', U, dm_prime, U.conj().T)
        dm_prime_next =(dm_prime_next + dm_prime_next.conj().T) / 2
        dm_ao_next = self.orth2ao(dm_prime_next, tar_type='dm')
        
        return dm_prime_next, dm_ao_next

    def prop_eular_step(self, time, step_size, dm_ao_lasthalf=None, dm_ao_this=None, tol=None, max_cycle=None): # dm input: dm_ao_this
        '''Euler propagator'''  # Only for testing
        step = np.round(time/step_size)
        
        dm_prime_next, dm_ao_next = self.elec_prop_step(time, step_size, dm_ao=dm_ao_this)
        
        fock_ao_next = self.get_fock(dm_ao_next, time+step_size)
       
        return (dm_prime_next, dm_ao_next, None, None), fock_ao_next
        
    def prop_mmut_step(self, time, step_size, dm_ao_lasthalf=None, dm_ao_this=None, tol=None, max_cycle=None): # dm input: dm_ao_this, dm_ao_lasthalf
        '''MMUT propagator''' #  https://doi.org/10.1039/B415849K
        step = np.round(time/step_size)
        
        dm_prime_nexthalf, dm_ao_nexthalf = self.elec_prop_step(time, step_size, dm_ao=dm_ao_lasthalf)
        dm_prime_next, dm_ao_next = self.elec_prop_step(time, 1/2 * step_size, dm_prime=dm_prime_nexthalf)
        
        fock_ao_next = self.get_fock(dm_ao_next, time+step_size)
        
        return (dm_prime_next, dm_ao_next, dm_prime_nexthalf, dm_ao_nexthalf), fock_ao_next
        
        
    def prop_lflp_pc_step(self, time, step_size, dm_ao_lasthalf=None, dm_ao_this=None, tol=None, max_cycle=None): # dm input: dm_ao_this，dm_ao_lasthalf
        '''LFLP-PC propagator''' #  https://doi.org/10.1063/1.5004675
        if tol is None:
            tol = 1e-7
        if max_cycle is None:
            max_cycle = 20
            
        step = np.round(time/step_size)
        
        # step 1
        dm_prime_this = self.ao2orth(dm_ao_this, tar_type='dm')
        fock_ao_this = self.get_fock(dm_ao_this, time)
        fock_prime_this = self.ao2orth(fock_ao_this, tar_type='fock')
        fock_ao_lasthalf = self.get_fock(dm_ao_lasthalf, time-1/2*step_size)
        fock_prime_lasthalf = self.ao2orth(fock_ao_lasthalf, tar_type='fock')
        
        fock_prime_nexthalf_p = 2  * fock_prime_this - fock_prime_lasthalf
        
        consistensy = False
        istep = 0
        while (not consistensy) and (istep <= max_cycle):  
            # step 2
            dm_prime_next, dm_ao_next = self.elec_prop_step(time, step_size, dm_prime=dm_prime_this, fock_prime=fock_prime_nexthalf_p)
            fock_ao_next = self.get_fock(dm_ao_next, time+step_size)
            # step 3
            dm_ao_nexthalf = (dm_ao_this + dm_ao_next) / 2
            dm_prime_nexthalf = (dm_prime_this + dm_prime_next) / 2
            # step 4
            fock_ao_nexthalf_c = self.get_fock(dm_ao_nexthalf, time+1/2*step_size)
            fock_prime_nexthalf_c = self.ao2orth(fock_ao_nexthalf_c, tar_type='fock')            
            # step 5
            consistensy = self.check_consistency(fock_prime_nexthalf_p, fock_prime_nexthalf_c, tol_=tol)
            if consistensy:
                return (dm_prime_next, dm_ao_next, dm_prime_nexthalf, dm_ao_nexthalf), fock_ao_next
            else:
                fock_prime_nexthalf_p = fock_prime_nexthalf_c 
                istep +=1       
        
    def prop_ep_pc_step(self, time, step_size, dm_ao_lasthalf=None, dm_ao_this=None, tol=None, max_cycle=None): # dm input: dm_ao_this
        '''EP-PC propagator''' #  https://doi.org/10.1063/1.5004675
        if tol is None:
            tol = 1e-7
        if max_cycle is None:
            max_cycle = 20
            
        step = np.round(time/step_size)
        
        # step 1
        dm_prime_this = self.ao2orth(dm_ao_this, tar_type='dm')
        fock_ao_this = self.get_fock(dm_ao_this, time)
        fock_prime_this = self.ao2orth(fock_ao_this, tar_type='fock')
        # step 2
        dm_prime_next_p, dm_ao_next_p = self.elec_prop_step(time, step_size, dm_prime=dm_prime_this, fock_prime=fock_prime_this)
             
        consistensy = False
        istep = 0
        while (not consistensy) and (istep < max_cycle):       
            # step 3
            fock_ao_next_p = self.get_fock(dm_ao_next_p, time+step_size)
            fock_prime_next_p = self.ao2orth(fock_ao_next_p, tar_type='fock')
            # step 4
            dm_prime_next_c, dm_ao_next_c = self.elec_prop_step(time, 1/2 * step_size, dm_prime=dm_prime_this, fock_prime=(fock_prime_this + fock_prime_next_p))
            # step 5
            consistensy = self.check_consistency(dm_prime_next_p, dm_prime_next_c, tol_=None)
            if consistensy:
                return (dm_prime_next_p, dm_ao_next_p, None, None), fock_ao_next_p
            else:
                dm_ao_next_p = dm_ao_next_c
                istep +=1
                   
    def kernel(self):
#         assert mf.converged is True, 'TD-SCF calculation must be initialized with a converged SCF result.'
        # need it for orthogonalization
        x = self.orth_ao(orth_method='lowdin')
        x_T = x.T
        x_inv = np.einsum('ji,jk->ik', x, self._scf.get_ovlp())
        x_inv_T = x_inv.T
        self.orth_xtuple = (x, x_T, x_inv, x_inv_T)
        
        # initialize propagation parameters
        assert self.max_step is not None, 'Please specify the maximum propagation step.'
        if self.step_size is None:
            self.step_size = 0.02
            print('No step size is given, default setting, %4.3f, will be used.'%self.step_size)
            
        if self.prop_method == 'eular':
            prop_step = self.prop_eular_step
            dm_ao_lasthalf_ = None
            dm_ao_this_ = self.dm_ao_init
        elif self.prop_method == 'mmut':
            prop_step = self.prop_mmut_step
            dm_ao_lasthalf_ = self.dm_ao_init
            dm_ao_this_ = self.dm_ao_init
        elif self.prop_method == 'eppc':
            prop_step = self.prop_ep_pc_step
            dm_ao_lasthalf_ = None
            dm_ao_this_ = self.dm_ao_init
        elif self.prop_method == 'lflp':
            prop_step = self.prop_lflp_pc_step
            dm_ao_lasthalf_ = self.dm_ao_init
            dm_ao_this_ = self.dm_ao_init
        else:
            error = True
            assert error is not True, 'No propagation method is specified or the specified method is not implemented'
            
        if self.save_step is None:
            self.save_step = 1
        
        time = 0.0
        
        self.ntime    = np.zeros([self.max_step // self.save_step +1])
        self.nfock_ao = np.zeros([self.max_step // self.save_step +1] + list(self.dm_ao_init.shape), dtype=np.complex128)
        self.ndm_ao   = np.zeros([self.max_step // self.save_step +1] + list(self.dm_ao_init.shape), dtype=np.complex128)
        self.netot    = np.zeros([self.max_step // self.save_step +1])
        
        self.ndm_ao[0]   = self.dm_ao_init
        self.nfock_ao[0] = self.get_fock(self.dm_ao_init, time)
        self.netot[0]    = self._scf.energy_tot(dm=self.dm_ao_init, h1e=self.get_hcore(), vhf=self.vhf_ao_init).real
        self.ntime[0]    = time
        
        # propagtation starts here
        istep = 0
        
        while istep < self.max_step:
            print('  Step of electronic propagation: ',istep+1)
            dms, fock_ao_next_ = prop_step(time, self.step_size, dm_ao_lasthalf=dm_ao_lasthalf_, dm_ao_this=dm_ao_this_, tol=None, max_cycle=100)
            
            dm_prime_next_       = dms[0]
            dm_ao_next_          = dms[1]
            dm_prime_nexthalf_   = dms[2]
            dm_ao_nexthalf_      = dms[3]           
            
            time += self.step_size
            
            if (istep / self.save_step) == (istep // self.save_step):
                self.ntime[istep // self.save_step + 1]    = time
                self.ndm_ao[istep // self.save_step + 1]   = dm_ao_next_
                self.nfock_ao[istep // self.save_step + 1] = fock_ao_next_
                self.netot[istep // self.save_step + 1]    = self._scf.energy_tot(dm=dm_ao_next_, h1e=self.get_hcore(), vhf=self._scf.get_veff(dm=dm_ao_next_)).real
            
            dm_ao_this_          = dm_ao_next_
            dm_ao_lasthalf_      = dm_ao_nexthalf_
           
            istep += 1
            
        
        # post-processing
        self.ndip    = np.zeros([self.max_step // self.save_step + 1,             3])
        self.npop    = np.zeros([self.max_step // self.save_step + 1, self.mol.natm])
        s1e = self._scf.get_ovlp()
        for i,idm in enumerate(self.ndm_ao):
            self.ndip[i] = self._scf.dip_moment(dm = idm.real, unit='au', verbose=0)
            self.npop[i] = self._scf.mulliken_pop(dm = idm.real, s=s1e, verbose=0)[1]
            
            
class EhrenDynamics(lib.StreamObject):
    def __init__(self,mf):
        self._scf                  = mf
        self.mol                   = mf.mol
        self.nd                    = NucDynamics(self._scf)
        
        self.nd_step_size          = None
        self.nd_sub_step_size      = None
        self.ed_step_size          = None
        
        self.nd_max_step           = None
        self.nd_sub_max_step       = None
        self.ed_max_step           = None
        
        self.ed_prop_method        = None
        self.save_step             = None
        
        self.num_time_scale        = 2
        
        self.dm_ao_init            = self._scf.make_rdm1()
        self.coord_init            = None
        self.momenta_init          = None
        self.force_init            = None
        self.vib_ene_init          = None
        self.ene_init              = None
        
    def check_setting(self):
        assert self.nd_max_step is not None, "Please specify 'nd_max_step'."
        assert self.nd_step_size is not None, "Please specify 'nd_step_size'."
        
        assert self.ed_max_step is not None, "Please specify 'ed_max_step'."
        assert self.ed_step_size is not None, "Please specify 'ed_step_size'."
                
        if self.num_time_scale == 2:
            assert self.nd_step_size // self.ed_step_size - self.ed_max_step < 0.0000001, "Please check 'ed_step_size' or 'ed_max_step'."
        elif self.num_time_scale == 3:
            assert self.nd_sub_max_step is not None, "Please specify 'nd_sub_max_step'."
            assert self.nd_sub_step_size is not None, "Please specify 'nd_sub_step_size'."
            assert self.nd_step_size // self.nd_sub_step_size - self.nd_sub_max_step < 0.0000001, "Please check 'nd_sub_step_size' or 'nd_sub_max_step'."
            assert self.nd_sub_step_size // self.ed_step_size - self.ed_max_step < 0.0000001, "Please check 'ed_step_size' or 'ed_max_step'."
        else:
            error = True
            assert error is False, "Invalid value for 'num_time_scale', please specify 2 or 3."
        
    def coord_prop_step(self, coord_this, momenta_this, force_this, step_size):        
        return self.nd.coord_prop_step(coord_this, momenta_this, force_this, step_size)
    
    def momenta_half_prop_step(self, momenta_nexthalf, force_next, step_size):
        return self.nd.momenta_half_prop_step(momenta_nexthalf, force_next, step_size)
    
    def orth_scratch(self):
        ed = ElecDynamics(self._scf) 
        
        x,x_inv =self.nd.orthogonalize(method='lowdin')
        x_T = x.T
        x_inv_T = x_inv.T
        self.orth_xtuple = (x, x_T, x_inv, x_inv_T)
        
    def elec_prop_init(self): 
        if self.ed_prop_method == 'eular' or self.ed_prop_method == 'eppc':                                             
            dm_ao_lasthalf_ = None
            dm_ao_this_ = self.dm_ao_init
        elif self.ed_prop_method == 'mmut' or self.ed_prop_method == 'lflp':
            dm_ao_lasthalf_ = self.dm_ao_init
            dm_ao_this_ = self.dm_ao_init
        else:
            error = True
            assert error is not True, 'No propagation method is specified or the specified method is not implemented'
        
        return dm_ao_this_, dm_ao_lasthalf_
    
    def elec_prop_step(self, time, step_size, dm_ao_lasthalf_, dm_ao_this_):
        ed = ElecDynamics(self._scf)
        ed.step_size = step_size
        ed.orth_xtuple = self.orth_xtuple
        if self.ed_prop_method == 'eular':
            prop_step = ed.prop_eular_step
        elif self.ed_prop_method == 'mmut':
            prop_step = ed.prop_mmut_step
        elif self.ed_prop_method == 'eppc':
            prop_step = ed.prop_ep_pc_step
        elif self.ed_prop_method == 'lflp':
            prop_step = ed.prop_lflp_pc_step
        else:
            error = True
            assert error is not True, 'No propagation method is specified or the specified method is not implemented'
            
        return prop_step(time, self.ed_step_size, dm_ao_lasthalf=dm_ao_lasthalf_, dm_ao_this=dm_ao_this_, tol=None, max_cycle=100)
            
    def update_mol(self, new_coord):
        self.mol = self.mol.set_geom_(new_coord * lib.param.BOHR, inplace=True)
        self._scf.reset(self.mol)
#         self._scf.kernel()
    
    def cal_force(self, dm_ao=None, dm_prime=None):
        if dm_ao is None and dm_prime is None:
            error = True
            assert error is not True, 'No dm is given.'
        elif dm_ao is not None and dm_prime is None:
            dm_ao = dm_ao
        elif dm_ao is None and dm_prime is not None:
            dm_ao = np.einsum('ij,jk,kl->il',self.orth_xtuple[0],dm_prime, self.orth_xtuple[1])

        new_force = self.nd.get_grad_elec(dm_ao) + self.nd.get_grad_nuc()
        
        return -new_force
            
    def ehrenfest_step_2t(self, time, coord_this, momenta_this, force_this, dm_ao_this, dm_ao_lasthalf, scf_max_cycle):
        '''One step of 2-time-scale Ehrenfest molecular dynamics procedure'''
        # Verlet step
        momenta_nexthalf, coord_next = self.coord_prop_step(coord_this, momenta_this, force_this, self.nd_step_size)
        
        # RT-TDHF step
        dm_ao_this, dm_ao_lasthalf = self.elec_prop_init()
        istep = 0
        
        time_e = time
        while istep < self.ed_max_step:
            dms, fock_ao_next_ = self.elec_prop_step(time_e, self.ed_step_size, dm_ao_lasthalf, dm_ao_this)
            
            dm_prime_next        = dms[0]
            dm_ao_next           = dms[1]
            dm_prime_nexthalf    = dms[2]
            dm_ao_nexthalf       = dms[3]   
            
            dm_ao_this           = dm_ao_next
            dm_ao_lasthalf       = dm_ao_nexthalf
            
            time_e += self.ed_step_size 
            istep += 1            

        self.update_mol(coord_next)
    
        self.orth_scratch()
        
        self.xtuple.append(self.orth_xtuple)
    
        dm_ao_next = np.einsum('ij,jk,kl->il',self.orth_xtuple[0],dm_prime_next, self.orth_xtuple[1])
        
        force_next = self.cal_force(dm_ao=dm_ao_next)
        momenta_next = self.momenta_half_prop_step(momenta_nexthalf, force_next, self.nd_step_size)
            
        return (coord_next, momenta_next, force_next), (dm_prime_next, dm_ao_next, dm_prime_nexthalf, dm_ao_nexthalf)   
    
    def ehrenfest_step_3t(self, time, coord_this, momenta_this, force_this, dm_ao_this, dm_ao_lasthalf, scf_max_cycle):
        '''One step of 3-time-scale Ehrenfest molecular dynamics procedure'''
        # Verlet step
        momenta_nexthalf, coord_next = self.coord_prop_step(coord_this, momenta_this, force_this, self.nd_step_size)
        
        # Mid-point step
        istep1 = 0
        time_Ne = time
        while istep1 < self.nd_sub_max_step:
            coord_nexthalf_sub = coord_this + self.nd.cal_velocity(momenta_nexthalf, self.nd.get_atom_mass(self.mol)) * ((istep1 + 1/2) * self.nd_sub_step_size)
            
#             print('3t:coord_nexthalf_sub:', coord_nexthalf_sub * lib.param.BOHR)
            dm_ao_this, dm_ao_lasthalf = self.elec_prop_init()
            
            dm_prime_this = np.einsum('ij,jk,kl->il',self.orth_xtuple[2],dm_ao_this, self.orth_xtuple[3])
            if dm_ao_lasthalf is not None:
                dm_prime_lasthalf = np.einsum('ij,jk,kl->il',self.orth_xtuple[2],dm_ao_lasthalf, self.orth_xtuple[3])
                        
            self.update_mol(coord_nexthalf_sub)
            self.orth_scratch()
            
            dm_ao_this_ = np.einsum('ij,jk,kl->il',self.orth_xtuple[0],dm_prime_this, self.orth_xtuple[1])
            if dm_ao_lasthalf is not None:
                dm_ao_lasthalf_ = np.einsum('ij,jk,kl->il',self.orth_xtuple[0],dm_prime_lasthalf, self.orth_xtuple[1])
        
            # RT-TDHF step
            
            istep2 = 0

            time_e = time_Ne
            while istep2 < self.ed_max_step:
                dms, fock_ao_next_ = self.elec_prop_step(time_e, self.ed_step_size, dm_ao_lasthalf_, dm_ao_this_)

                dm_prime_next        = dms[0]
                dm_ao_next           = dms[1]
                dm_prime_nexthalf    = dms[2]
                dm_ao_nexthalf       = dms[3]   

                dm_ao_this_           = dm_ao_next
                dm_ao_lasthalf_       = dm_ao_nexthalf

                time_e += self.ed_step_size 
                istep2 += 1 
            
            self.dm_ao_init = dm_ao_next
            time_Ne += self.nd_sub_step_size
            istep1 += 1

        self.update_mol(coord_next)    
        self.orth_scratch()
        
        self.xtuple.append(self.orth_xtuple)
    
        dm_ao_next = np.einsum('ij,jk,kl->il',self.orth_xtuple[0],dm_prime_next, self.orth_xtuple[1])
        
        force_next = self.cal_force(dm_ao=dm_ao_next)
        momenta_next = self.momenta_half_prop_step(momenta_nexthalf, force_next, self.nd_step_size)
            
        return (coord_next, momenta_next, force_next), (dm_prime_next, dm_ao_next, dm_prime_nexthalf, dm_ao_nexthalf)  

    
    def vib_ene2momenta(self):
        mass = np.array([self.nd.get_atom_mass(self.mol)])
        eV2au = 0.036749405469679
        energy = self.vib_ene_init
        energy = energy * eV2au

        # Convert vibration energy (eV) to momenta (Only for 2-atom molecule)
        # Initially momenta is conserved: P_a = -P_b
        
        mass_r = (mass[0][0] + mass[0][1]) / (mass[0][0] * mass[0][1])
        vector = self.mol.atom_coords()[0] - self.mol.atom_coords()[1]
        direction = vector / (np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2))
        momenta_sq = energy / mass_r * 2 
        momenta = np.sqrt(momenta_sq)
        self.momenta_init = -momenta * np.array((direction,-direction))

        
    def momenta2ene(self, momenta):
        au2ev = 27.211386246
        mass = np.array([self.nd.get_atom_mass(self.mol)])
        mass_r = (mass[0][0] + mass[0][1]) / (mass[0][0] * mass[0][1])
        ene = np.linalg.norm(momenta[0])**2 * mass_r / 2
        
        return ene
    
    def kernel(self, scf_max_cycle=None):
        '''Ehrenfest molecular dynamics procedure'''
        self.check_setting()
        
        dm_ao_this = self.dm_ao_init
        dm_ao_lasthalf = None
#         print('step 0', dm_ao_this)
        
        self.orth_scratch()
        
        if self.vib_ene_init is not None:
            self.vib_ene2momenta()
        
        if self.save_step is None:
            self.save_step = 1 

        if self.coord_init is None:
            coord_this = self.mol.atom_coords()
        else:
            coord_this = self.coord_init
            
        if self.momenta_init is None:
            momenta_this = np.zeros(list(coord_this.shape), dtype=np.float64)
            self.momenta_init = momenta_this
        else:
            momenta_this = self.momenta_init
            
        if self.force_init is None:
            force_this = self.cal_force(dm_ao=dm_ao_this)
        else:
            force_this = self.force_init
            
        if self.num_time_scale == 2:
            ehrenfest_step = self.ehrenfest_step_2t
        elif self.num_time_scale == 3:
            ehrenfest_step = self.ehrenfest_step_3t
        
        time = 0.0
        etot_this = self._scf.e_tot
        
        self.nforces   = np.zeros([self.nd_max_step // self.save_step + 1] + list(coord_this.shape), dtype=np.float64)
        self.nmomenta  = np.zeros([self.nd_max_step // self.save_step + 1] + list(coord_this.shape), dtype=np.float64)
        self.ncoords   = np.zeros([self.nd_max_step // self.save_step + 1] + list(coord_this.shape), dtype=np.float64)
        self.ntime     = np.zeros([self.nd_max_step // self.save_step + 1])
        self.netot     = np.zeros([self.nd_max_step // self.save_step + 1])
        self.nE_tot    = np.zeros([self.nd_max_step // self.save_step + 1])
        self.ndm_ao    = np.zeros([self.nd_max_step // self.save_step + 1] + list(self.dm_ao_init.shape), dtype=np.complex128)
        self.ndm_prime = np.zeros([self.nd_max_step // self.save_step + 1] + list(self.dm_ao_init.shape), dtype=np.complex128)
        
        self.nforces[0]   = force_this
        self.nmomenta[0]  = momenta_this
        self.ncoords[0]   = coord_this
        self.ntime[0]     = time
        self.ndm_ao[0]    = dm_ao_this
        self.ndm_prime[0] = np.einsum('ij,jk,kl->il',self.orth_xtuple[2],dm_ao_this, self.orth_xtuple[3])
        self.netot[0]     = self._scf.energy_tot(dm=dm_ao_this, h1e=self._scf.get_hcore(), vhf=self._scf.get_veff(dm=dm_ao_this)).real
        self.nE_tot[0]    = self.momenta2ene(self.momenta_init) + self.netot[0]
        self.xtuple = []
        self.xtuple.append(self.orth_xtuple)
            
        istep = 1
        while istep < self.nd_max_step+1:
            print('Step of Ehtenfest dynamics: ', istep)
            motions, dms = ehrenfest_step(time, coord_this, momenta_this, force_this, dm_ao_this, dm_ao_lasthalf, scf_max_cycle)
            coord_next      = motions[0]
            momenta_next    = motions[1]
            force_next      = motions[2]
            
            dm_prime_next     = dms[0]
            dm_ao_next        = dms[1]
            dm_prime_nexthalf = dms[2]
            dm_ao_nexthalf    = dms[3]

            time += self.nd_step_size
            
            # save
            if (istep / self.save_step) == (istep // self.save_step):
#                 print('dm_ao of saved step %d: \n'%(istep//self.save_step), dm_ao_next)  # need to be replaced with "log" function -- 2020/10/22
                self.nforces[istep//self.save_step]   = force_next
                self.nmomenta[istep//self.save_step]  = momenta_next
                self.ncoords[istep//self.save_step]   = coord_next
                self.ntime[istep//self.save_step]     = time
                self.netot[istep//self.save_step]     = self._scf.energy_tot(dm=dm_ao_next, h1e=self._scf.get_hcore(), vhf=self._scf.get_veff(dm=dm_ao_next)).real
                self.nE_tot[istep//self.save_step]    = self.momenta2ene(self.nmomenta[istep//self.save_step]) + self.netot[istep//self.save_step]
                self.ndm_ao[istep//self.save_step]    = dm_ao_next
                self.ndm_prime[istep//self.save_step] = dm_prime_next
            
            # update
            coord_this      = coord_next
            force_this      = force_next
            momenta_this    = momenta_next
            
            dm_ao_this      = dm_ao_next
            dm_lasthalf     = dm_ao_nexthalf
            self.dm_ao_init = dm_ao_next
#             print('kernel:final dm_prime of %d:\n'%istep, np.einsum('ij,jk,kl->il',self.orth_xtuple[2],dm_ao_this, self.orth_xtuple[3]),'\n\n')
            
            istep +=1
    