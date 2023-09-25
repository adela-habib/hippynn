import ase.io.ulm as ulm
import numpy as np
import os
import sys

Z = 47
n_atoms = 147
dirNums = np.arange(200,318) 
print(dirNums)
totalDir = dirNums[-1]-dirNums[0]+1 
tStop = 1600 
n_basisfunc = 18
nBands_KS = 1200
i0, iN, a0, aN = 0, 303, 300, 989
nBands = aN+1 #specified in the time-dependent calculation
nBands_a = nBands_KS-a0
features_e_AO = np.zeros((totalDir, n_atoms, n_basisfunc, tStop))
features_h_AO = np.zeros((totalDir, n_atoms, n_basisfunc, tStop))
features_e_KS = np.zeros((totalDir, nBands_KS, tStop))
features_h_KS = np.zeros((totalDir, nBands_KS, tStop))
for iEdir in dirNums:
    h_t_iEdir_AO = np.load('Edir-'+str(iEdir)+'/'+ str(iEdir) + "-h-overTime-AO.npy")
    e_t_iEdir_AO = np.load('Edir-'+str(iEdir)+'/'+ str(iEdir) + "-e-overTime-AO.npy")
    h_t_iEdir_KS = np.load('Edir-'+str(iEdir)+'/'+ str(iEdir) + "-h-overTime-KS.npy")
    e_t_iEdir_KS = np.load('Edir-'+str(iEdir)+'/'+ str(iEdir) + "-e-overTime-KS.npy")
    h_t_iEdir_AO = h_t_iEdir_AO.reshape(n_atoms, n_basisfunc, tStop)
    e_t_iEdir_AO = e_t_iEdir_AO.reshape(n_atoms, n_basisfunc, tStop)
    P_n_h_KS = np.zeros((nBands, tStop))
    P_n_e_KS = np.zeros((nBands, tStop))
    P_n_h_KS[:iN+1, :] = h_t_iEdir_KS
    P_n_e_KS[a0:, :] = e_t_iEdir_KS[:nBands_a, :]
    features_h_AO[iEdir-dirNums[0], :, :, :] = h_t_iEdir_AO
    features_e_AO[iEdir-dirNums[0], :, :, :] = e_t_iEdir_AO
    features_h_KS[iEdir-dirNums[0], :, :] = P_n_h_KS
    features_e_KS[iEdir-dirNums[0], :, :] = P_n_e_KS
    print("Working on the ", iEdir,"\n")

outFileName = 'Ag147-all-'
np.save(outFileName+'h-AO.npy', features_h_AO)
np.save(outFileName+'e-AO.npy', features_e_AO)
np.save(outFileName+'h-KS.npy', features_h_KS)
np.save(outFileName+'e-KS.npy', features_e_KS)
