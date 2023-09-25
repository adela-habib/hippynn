import torch
torch.set_default_dtype(torch.float64)
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import timeit
import hippynn
from hippynn.experiment.serialization import load_checkpoint
from hippynn.graphs import Predictor
torch.set_default_dtype(torch.float64); 
print("torch opm num thread:", torch.get_num_threads())
torch.set_num_threads(32)
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

if len(sys.argv) < 9:
    print('python MLCharge.py <nhist> <dt> <training Model dirName> <outFileName> <n atoms> <iEdir> <scale> <recurrModel True (1) or False (0)>' )
    exit()

nhist = int(sys.argv[1])
dt = int(sys.argv[2])
trainModelDir = sys.argv[3]
outFilename = sys.argv[4]
n_atoms = int(sys.argv[5])
sample = sys.argv[6]
scale = float(sys.argv[7])
recurrModel = int(sys.argv[8])
nSteps = 1300 # number of time steps after pulse with the first as t0
tStep = 20 # attoseconds
Z = 47
pulseEndTimeStep = 6000  #attoseconds
num_Int = nSteps-(nhist-1)
QMdata = "../data-gpaw/"

#Load model from multi-step training
if recurrModel == 1:
    checkpoint = torch.load(trainModelDir+'/checkpoint.pth', map_location="cpu")
    model = checkpoint['model']
else:
    bundle = load_checkpoint(trainModelDir+'/experiment_structure.pt',
                             trainModelDir+'/best_checkpoint.pt',
                             restore_db=False, map_location = torch.device('cpu'))
    model = bundle["training_modules"].model


def run_inference_loop(myModel, iEdir):
    
    #Read ground state data and first few steps of rt-TDDFT
    coord = np.load(QMdata+'Ag'+str(n_atoms)+'-atom-pos.npy')
    GAP_t0 = np.load(QMdata+'Ag'+str(n_atoms)+"-GAP_t0.npy")
    GAP_t = np.load(QMdata+iEdir+"-MullikenElectronPop.npy").T #atoms, timeSteps
    trueDataAllTraj = True   #if we are testing true vs. ML for all times in a trajectory

    #Remove the pulse region and make delta
    afterPulse = np.arange(0,pulseEndTimeStep,tStep).shape[0]; #get the index of the pulse stop
    GAP_t = scale*(GAP_t[:, afterPulse:] - GAP_t0[:, None]) #scale the data
    GAP_t = torch.from_numpy(GAP_t)
    idataStop = nhist-1
    idataStart = 0

    #Make inputs for the first rt-TDDFT steps  
    coord_cur = torch.unsqueeze(torch.from_numpy(coord), 0)
    spec_cur = torch.from_numpy(np.ones((1, n_atoms), dtype=np.int64))*Z 
    traj_cur = torch.zeros([1, GAP_t.shape[0], nhist-1])
    traj_cur[0, :, :] = GAP_t[:, idataStart:idataStop]; 
    masterFile = np.zeros((n_atoms, nSteps))
    masterFile[:,:idataStop] = GAP_t[:,:idataStop]
    pred_output = Predictor.from_graph(myModel, return_device=device, model_device=device)
    
    #If the whole trajectory is given for loss calculation purposes 
    loss_RMSE = []
    loss_MAE = []
    
    #Make next steps predictions in the trajectory
    start = timeit.default_timer()
    for i in range(num_Int-1): 
       
        #New predicted values
        outputs = pred_output(coordinates=coord_cur, dfeatures=traj_cur, species=spec_cur) 
        ML_pred = outputs['HDensityPred.atom_charges']
       
        #Compute loss term for this step if true data given
        if trueDataAllTraj:
            #Values in the next time point in target
            target_x = torch.unsqueeze(GAP_t[:,idataStop], 1)
            error_RMSE = ((ML_pred-target_x)**2).mean().sqrt()
            error_MAE = (ML_pred-target_x).abs().mean()

            #Track losses
            loss_RMSE.append(error_RMSE.item())
            loss_MAE.append(error_MAE.item())

        #Add predicted to the master file
        masterFile[:, idataStop] = ML_pred.cpu().detach().numpy()[0,:,0]
        idataStop += 1

        #Add the predicted values into the input
        traj_cur = torch.cat([traj_cur[:,:,1:], ML_pred], dim=2) 

        #Report progress
        if i%100 == 0:
            print("Interval progress %:", np.ceil(100*(i/(num_Int-1))), idataStop)
    stop = timeit.default_timer()
    print('Total Time(s): ', stop - start)
    np.save('../data-infer/'+iEdir+'-'+str(nhist)+outFilename+'-MLCharge.npy', masterFile)
    
    return masterFile, GAP_t.detach().numpy(),loss_RMSE, loss_MAE


for iEdir in ['Ag'+str(n_atoms)+'-'+sample]:
    predQ, trueQ, loss_RMSE, loss_MAE = run_inference_loop(model, iEdir)
    print("avg. error: {:0.3f}, {:0.3f}".format(np.mean(np.array(loss_RMSE)), np.mean(np.array(loss_MAE))))
