import numpy as np
import os
import sys


if len(sys.argv) < 4:
    print("python make_ml_train_interval.py <nhist> <dt steps> <scale factor> ")
    exit()


nhist = int(sys.argv[1])
dt = int(sys.argv[2])
scale = float(sys.argv[3])
Z = 47
def getFeaturesAgNC(alltrajfName, n_trajThisCluster):

    all_traj = scale*np.load(alltrajfName+'-all-delta-el-EmagEdirTest.npy')/1000 
    atom_pos = np.load(alltrajfName+'-atom-pos.npy')
    all_traj = np.swapaxes(all_traj.reshape((n_trajThisCluster, tSteps, -1)), 1,2)
    n_atomsThisCluster = all_traj.shape[1]
    print(all_traj.shape)
    outData_features = np.zeros((n_trajThisCluster, ntimepoints, nAtomsMax, nhist-1))
    outData_targets = np.zeros((n_trajThisCluster, ntimepoints, nAtomsMax, 1))
    outData_species = np.zeros((n_trajThisCluster, ntimepoints, nAtomsMax), dtype=np.int64)
    outData_coord = np.zeros((n_trajThisCluster, ntimepoints, nAtomsMax, 3))

    for itraj in range(n_trajThisCluster):
        idataStop = dt*(nhist-1)
        for itime in range(ntimepoints):
            for ihist in range(nhist-1):
                idataStart = idataStop - (nhist-ihist-1)*(dt)
                outData_features[itraj, itime, :n_atomsThisCluster, ihist] = all_traj[itraj, :, idataStart]; print(alltrajfName, itraj, itime, idataStart, idataStop)

            outData_targets[itraj, itime, :n_atomsThisCluster, 0] = all_traj[itraj, :,idataStop]
            idataStop += (nhist*dt)
            outData_species[itraj, itime, :n_atomsThisCluster] = np.ones(n_atomsThisCluster)*Z
            outData_coord[itraj, itime, :n_atomsThisCluster, :] = atom_pos
    return outData_features, outData_targets, outData_species, outData_coord

tSteps = 1300 #after pulse 
ntimepoints = tSteps//(dt*nhist)
fName = ['Ag55', 'Ag147', 'AgOct146']
nAtomsMax = 147
ntrajNC = np.array([100, 100, 100])

totalTrajectories = ntrajNC.sum()

features_55, targets_55, species_55, coord_55 = getFeaturesAgNC(fName[0], ntrajNC[0])
features_147, targets_147, species_147, coord_147 = getFeaturesAgNC(fName[1], ntrajNC[1])
features_146, targets_146, species_146, coord_146 = getFeaturesAgNC(fName[2], ntrajNC[2])

totalOutfeatures = np.concatenate( (features_55, np.concatenate((features_146, features_147), axis=0)) , axis=0)
totalOutTargets = np.concatenate( (targets_55, np.concatenate((targets_146, targets_147), axis=0)), axis=0)
totalOutspecies = np.concatenate( (species_55, np.concatenate((species_146, species_147), axis=0)), axis=0)
totalOutcoord = np.concatenate( (coord_55, np.concatenate((coord_146, coord_147), axis=0)), axis=0)

totalOutfeatures = totalOutfeatures.reshape(totalTrajectories*ntimepoints, nAtomsMax, nhist-1)
totalOutTargets = totalOutTargets.reshape(totalTrajectories*ntimepoints, nAtomsMax, 1)
totalOutspecies = totalOutspecies.reshape(totalTrajectories*ntimepoints, nAtomsMax)
totalOutcoord = totalOutcoord.reshape(totalTrajectories*ntimepoints, nAtomsMax, 3)
print(np.where(totalOutspecies==0)[0].shape)
print(totalOutTargets.shape, totalOutcoord.shape, totalOutspecies.shape, totalOutfeatures.shape)

fileNameSuffix = 'Ag-'+str(nhist) + '-' + str(int(scale))+'x'
np.save(fileNameSuffix+'-dfeatures.npy', totalOutfeatures)
np.save(fileNameSuffix+'-dtargets.npy', totalOutTargets)
np.save(fileNameSuffix+'-species.npy', totalOutspecies)
np.save(fileNameSuffix+'-coordinates.npy', totalOutcoord)
