import os
import sys
import numpy as np
import ase.io.ulm as ulm
from gpaw.lcaotddft import LCAOTDDFT

def getRhoOccTime(filename, overlap):
    """
    input: ground state gpw file
    output: density matrix and occu factor at that time
    """
    #Calculate density matrices from the overlap and wavefunctions
    wfn = ulm.Reader(filename) #at one instance of time for now
    wf = np.array(wfn.wave_functions.coefficients[0,0,:,:])  # (a,b,c for band indices)nband, orbitals (mu)
    occ = np.array(wfn.wave_functions.occupations[0,0,:]) #nbands
    #Normalize the wavefunctions (GPAW LCAO basis set is not normalized)
    wfj = np.conj(wf).T #nu, nbands
    norm_factor = np.dot(wf,np.dot(overlap,wfj))[0,0] #
    wf /= np.sqrt(np.abs(norm_factor))
    wfj /= np.sqrt(np.abs(norm_factor))
    rhoT = np.dot(wf.T,(wfj*occ).T)
    return rhoT, occ

def getElpopPerAtom(filename, overlap):
    """
    Input: wavefunction at an instant of time
    Out: numpy 1D array of gross atom population in Mulliken Analysis
    """
    rhoT, _ = getRhoOccTime(filename, overlap)
    popT = np.dot(rhoT.real,overlap.real)
    print('Valence electrons: ', np.trace(popT)*2)
    pop_orbital = popT.diagonal()*2
    GAP_i = pop_orbital.reshape(n_atoms, -1).sum(axis=1) #Gross atom population summer over basis functions belonging to atom i
     #assuming axis=1 holds number of basis functions per atom?'''
    return GAP_i 

def getElPopPerAtomPerTime(stepStart, stepEnd, natoms, overlap):
    """
    Input: the # of time steps and # of atoms in the cluster
    Output: numpy 2D array of gross atom population per atom per time
    """
    nSteps = stepEnd - stepStart
    outGAP_i = np.zeros((nSteps, natoms))
    for istep in range(stepStart, stepEnd):
        filename = "wf-" +str(istep+1).zfill(6)+ ".ulm"
        x, y = getRhoOccTime(filename, overlap) 
        outGAP_i[istep-stepStart] = getElpopPerAtom(filename, overlap)
        print("Time step:", istep)
    return outGAP_i

#User specified inputs/args

irun = sys.argv[1]
stepSt = int(sys.argv[2]) #td.tddft.niter
stepEn = int(sys.argv[3])

tstep = 20 #attoseconds
pulseEndTimeStep = 6000
#Read binary files for input data
#td_calc = LCAOTDDFT('gs.gpw')
overlap = np.load('overlap.npy') #np.array(td_calc.wfs.kpt_u[0].S_MM) # mu, nu one k-point in the calc so kpt_u[0]
#np.save('overlap.npy', overlap)
td = ulm.Reader('td.gpw')
n_atoms = td.atoms.positions.shape[0]
atom_pos = np.array(td.atoms.positions)
#np.save('atom_pos.npy', atom_pos)
GAP_t0 = np.load('GAP_t0.npy') #getElpopPerAtom('gs.gpw', overlap)
#np.save('GAP_t0.npy', GAP_t0)
mul_pop = getElPopPerAtomPerTime(stepSt, stepEn, n_atoms, overlap)
np.save('MullikenElectronPop-'+irun+'.npy', mul_pop)
