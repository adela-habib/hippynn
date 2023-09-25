import numpy as np
carrier = ""
nparts = 17
natoms = 201
ntSteps = 1600

save_pop1 = np.load('MullikenElectronPop-'+carrier+'1.npy')
for ipart in range(2, nparts):
    print(ipart)
    iPop = np.load('MullikenElectronPop-'+carrier+str(ipart)+'.npy'); print(save_pop1.shape, iPop.shape)
    save_pop1 =  np.hstack((save_pop1.T, iPop.T)).T
print('total pop file shape', save_pop1.shape)
np.save('MullikenElectronPop-'+carrier+'.npy', save_pop1)
