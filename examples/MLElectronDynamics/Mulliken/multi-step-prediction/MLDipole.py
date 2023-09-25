import sys

if len(sys.argv) < 6:
    print('python MLDipole.py <nhist> <outFileName> <n atoms> <iEdir> <scale>' )
    print('For example: python MLDipole.py 4 nhis4-single-step 55 219 100')
    exit()
    
    
import numpy as np
nhist = int(sys.argv[1])
outFilename = sys.argv[2]
n_atoms = int(sys.argv[3])
sample = sys.argv[4]
scale = float(sys.argv[5])
AgNC = 'Ag'+str(n_atoms)
iEdir = AgNC +'-'+sample
dtStep = 20 #20 attoseconds
pulseEndTimeStep = 6000  #in intervals of 20 attoseconds
totalTimesteps = 1600
afterPulseIndex = 300
nSteps = totalTimesteps - afterPulseIndex
num_Int = nSteps-(nhist-1)
QMdata = "../data-gpaw/"
dataInferDir = '../data-infer/'
Z = 47
Z_i = 11 #number of valence electrons per atom in the simulation


#Calculate dipole moments given charges and atomic positions
def getDipoleMom(atom_pos, el_charge):
    cent_v = atom_pos-atom_pos.mean(axis=0)
    if len(el_charge.shape) > 1:
        print("Total number of electrons per timestep:", np.sum(el_charge)/el_charge.shape[0])
    else:
        print("Total number of electrons:", np.sum(el_charge))
    dipoleMom = np.dot((Z_i-el_charge), cent_v)
    return dipoleMom

#Make dynamic dipole moment plots 
def plotMLDM(iEdir):
    
    #Read input files
    atom_pos = np.load(QMdata+AgNC+'-atom-pos.npy') 
    dm_Edir = np.loadtxt(QMdata+iEdir+'-dm.dat')[:, :]
    gpaw_Edir = np.load(QMdata+iEdir+'-MullikenElectronPop.npy')
    gpaw_t0 = np.load(QMdata+AgNC+'-GAP_t0.npy')
    predicted_charges = np.load(dataInferDir + dataInferDir+iEdir+'-'+str(nhist)+outFilename+'-MLCharge.npy')[:int(AgNC[2:]),:]
    
    #Calculate DM
    dm_MPAt0 = getDipoleMom(atom_pos, (gpaw_t0))
    dm_MPA = getDipoleMom(atom_pos, (gpaw_Edir-gpaw_t0[None, :]))[afterPulseIndex:, :]    
    dm_ml = getDipoleMom(atom_pos, predicted_charges.T/scale)
    np.save(dataInferDir+iEdir+'-'+str(nhist)+outFilename+'-MLdm.npy', dm_ml)
    
    #Plot DM over time
    dirLab = ['x', 'y', 'z']
    timeGPAW = np.arange(0,dm_Edir.shape[0]*dtStep,dtStep)/1000 #in fs
    timeML = np.arange(afterPulseIndex*dtStep,totalTimesteps*dtStep,dtStep)/1000 #in fs
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.,2.5), dpi= 80, facecolor='w', edgecolor='k')
    gs = fig.add_gridspec(1,3, hspace=0.1, wspace=0.3)
    plotStop = 1300
    for idir in range(3):
        ax1 = fig.add_subplot(gs[0, idir])
        ax1.set_title("Dir. "+dirLab[idir])
        ax1.plot(2*dm_ml[:plotStop,idir], 'b', label=dirLab[idir]+' ML')
        ax1.plot(dm_Edir[afterPulseIndex:afterPulseIndex+plotStop,2+idir], 'k--', label=dirLab[idir]+' GPAW')
        ax1.set_xlabel('Time steps (each 20 attoseconds)')
        if idir == 0:
            ax1.set_ylabel('Dipole moment (|e|A)')
    plt.tight_layout()
    plt.show()
    return timeGPAW, dm_Edir[:,2:], dm_MPA, dm_ml, predicted_charges, gpaw_Edir, gpaw_t0 
 

time, dm_gpaw, dm_MPA, dm_ML, pred_Q, true_Q, true_t0_Q = plotMLDM(iEdir)
