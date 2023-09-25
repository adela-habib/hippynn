import numpy as np
import ase.io.ulm as ulm
from gpaw.lcaotddft import LCAOTDDFT
import matplotlib.pyplot as plt
import os
from subprocess import Popen, PIPE

def getRhoOcc_perTime(filename, overlap):
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
    return wf, wfj, rhoT, occ

#Read binary files for input data
overlap = np.load('overlap.npy')

#Othogonalize e-h
E, Evec = np.linalg.eigh(overlap)
invSqrtOverlap = np.dot(Evec, np.dot(np.diag(1./np.sqrt(E)), Evec.conj().T))
sqrtOverlap = np.dot(Evec, np.dot(np.diag(np.sqrt(E)), Evec.conj().T))
testOrthog = np.dot(invSqrtOverlap.conj().T, np.dot(overlap, invSqrtOverlap))
print(testOrthog[:3,:3], E[:2])

#KS decomposition
from gpaw import GPAW
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.tddft.units import au_to_eV

# Calculate ground state with full unoccupied space
calc = GPAW('unocc.gpw', txt=None)#.fixed_density(nbands='nao', txt='unocc.out')
#calc.write('unocc.gpw', mode='all')

# Construct KS electron-hole basis
ksd = KohnShamDecomposition(calc)
ksd.initialize(calc, min_occdiff=1e-1, only_ia = True)
ksd.write('ksd.ulm')

#Get KSD characteristics
eig_n, fermilevel = ksd.get_eig_n(zero_fermilevel=True)
i_state = np.arange(ksd.ialims()[0], ksd.ialims()[1]+1)
a_state = np.arange(ksd.ialims()[2], ksd.ialims()[3]+1)
eig_n_i = eig_n[i_state]
eig_n_a = eig_n[a_state]
i0, iN, a0, aN = ksd.ialims()
nBands = 1200 #specified in the time-dependent calculation
nBands_a = nBands-a0
gs_bands = ulm.Reader('gs.gpw')
wfn0, wfnj0, rho_munu0, occ0 = getRhoOcc_perTime('unocc.gpw', overlap)
Umat = np.dot(wfn0, sqrtOverlap).real
print("Determinant of the rotation matrix:", np.linalg.det(Umat))
np.save('Umat.npy', Umat)
proj_matFull = Umat
proj_matInvFull = Umat.conj().T
proj_mat = proj_matFull[:nBands, :].T
proj_matInv = proj_matInvFull[:, :nBands].T

#np.save('sqrtOverlap.npy', sqrtOverlap)

def get_eh_over_time(nao, ntSteps):
    
    P_h_munu_t = np.zeros((nao, ntSteps))
    P_e_munu_t = np.zeros((nao, ntSteps))
    
    P_h_i_t = np.zeros((i_state.shape[0], ntSteps))
    P_e_a_t = np.zeros((a_state.shape[0], ntSteps))    
    
    for istep in range(ntSteps):
        print("Working in time step", istep, "...")
        filename = "wf-" +str(istep+1).zfill(6)+ ".ulm"
        wfn, wfnj, rho_munu, occ = getRhoOcc_perTime(filename, overlap)
        drho_munu = rho_munu-rho_munu0 #drho in lcao space
        
        #e-h KS basis
        rho_up1 = ksd.transform([drho_munu]) #rho in i-a pair but flattened
        P_ia_t_flat = np.abs(rho_up1[0]/(np.sqrt(ksd.f_p)))**2 / EfieldStrength  ## in a flattened array
        P_ia_t = ksd.M_p_to_M_ia(P_ia_t_flat) #form a 2D array here
        P_i_h = np.sum(P_ia_t, axis=1)  
        P_a_e = np.sum(P_ia_t, axis=0)
        P_h_i_t[:,istep] = P_i_h
        P_e_a_t[:,istep] = P_a_e
        
        #For the case when nBands KS is not equal to and to match the full KS bands
        P_n_h_KS = np.zeros(nBands)
        P_n_e_KS = np.zeros(nBands)
        P_n_h_KS[:iN+1] = P_i_h
        P_n_e_KS[a0:] = P_a_e[:nBands_a]
        #Convert from KS to AO
        P_h_munu = np.dot(proj_mat, P_n_h_KS)
        P_e_munu = np.dot(proj_mat, P_n_e_KS) 
        #To test transformation convert AO back to KS
        P_h_i_trans = np.dot(proj_matInv, P_h_munu)
        P_e_a_trans = np.dot(proj_matInv, P_e_munu)

        print('sum of prob. of h and e in gpaw KS: {:0.5f}, {:0.5f}'.format(np.sum(P_n_h_KS), np.sum(P_n_e_KS)))
        print('sum of prob. of h and e in AO: {:0.5f}, {:0.5f}'.format(np.sum(np.abs(P_h_munu)), np.sum(np.abs(P_e_munu))))
        print('sum of prob. of h and e in KS tranformed from AO: {:0.5f}, {:0.5f}'.format(np.sum(P_h_i_trans), np.sum(P_e_a_trans)))

        P_h_munu_t[:,istep] = P_h_munu
        P_e_munu_t[:,istep] = P_e_munu
            
    np.save(iEdir + "-h-overTime-AO.npy", P_h_munu_t)
    np.save(iEdir + "-e-overTime-AO.npy", P_e_munu_t)
    np.save(iEdir + "-h-overTime-KS.npy", P_h_i_t)
    np.save(iEdir + "-e-overTime-KS.npy", P_e_a_t)
    
    return P_h_munu_t, P_e_munu_t, P_h_i_t, P_e_a_t

COMMAND = '''awk ' $1=="Emag" {print $3}' td-time-Emag.py'''
EfieldStrength = float(Popen(COMMAND, shell=True, stdout=PIPE, stderr=PIPE).communicate()[0].strip()) 
td = ulm.Reader('td.gpw')
n_atoms = td.atoms.positions.shape[0]
n_steps = 1600 #td.tddft.niter
iEdir = os.environ['iEdir']
nao = n_atoms*18 #18 is basis functions per atom 
h_AO, e_AO, h_KS, e_KS = get_eh_over_time(nao, n_steps)

