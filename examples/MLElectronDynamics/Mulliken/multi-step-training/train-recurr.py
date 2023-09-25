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
from hippynn.databases import DirectoryDatabase

if len(sys.argv) < 11:
    print("""train-recurr-randData.py <gpu device #> <# of epochs> <# fragments> <nfactor> <outFileDir> <init model from HIPNN> <nhist> <batch_size> <factorForSample> <TAG 0 fresh run and 1 restart> <init_lr> <scale data>""")
    exit()

#Variable parameters
gpuDevice = sys.argv[1]
n_iterations = int(sys.argv[2])
n_recurr = int(sys.argv[3])
outFileDir = sys.argv[4]
trainModelDir = sys.argv[5] #'../nhis4-1e2conserveQ-1IL'
nhist = int(sys.argv[6])
batch_size = int(sys.argv[7])
nfactor = int(sys.argv[8])
TAG = int(sys.argv[9])
init_lr = float(sys.argv[10])
scaleData = float(sys.argv[11])
print("GPU device #:", gpuDevice)
print("nRecurr:", n_recurr)
print("nhist:", nhist)
print("batch size:", batch_size)
print("n random sampling:", nfactor)
print("TAG for starting (0) or restarting (1):", TAG)
print("Initial learning rate:", init_lr)
print("Scaled data with:", str(scaleData), "x")
dataDir = '../data/'

#Constants
ntraj55 = 100
ntraj147 = 100
ntraj146 = 100
ntraj = ntraj55 + ntraj147 + ntraj146
nSys = int(nfactor*ntraj)
ntimePoints = 1300
natoms = 147
splitpc = 0.2
nhist_here = n_recurr+20
Z = 47
threshold = 1e-8
duration = 0
start1 = timeit.default_timer() 
patienceVal = 20


if not os.path.exists(outFileDir):
    os.mkdir(outFileDir)
else:
    pass
if torch.cuda.is_available():
    device = 'cuda:'+gpuDevice
else:
    device = 'cpu'

#Read input data files
coordinates55 = np.load(dataDir+'Ag55-atom-pos.npy')
coordinates146 = np.load(dataDir+'AgOct146-atom-pos.npy')
coordinates147 = np.load(dataDir+'Ag147-atom-pos.npy')
masterData147 = scaleData*np.load(dataDir+'Ag147-all-delta-el-EmagEdirTest.npy')/1000
masterData147 = np.swapaxes( masterData147.reshape((ntraj147, ntimePoints, 147)), 1,2)
masterData55 = scaleData*np.load(dataDir+'Ag55-all-delta-el-EmagEdirTest.npy')/1000
masterData55 = np.swapaxes( masterData55.reshape((ntraj55, ntimePoints, 55)), 1,2)
masterData146 = scaleData*np.load(dataDir+'AgOct146-all-delta-el-EmagEdirTest.npy')/1000
masterData146 = np.swapaxes( masterData146.reshape((ntraj146, ntimePoints, 146)), 1,2)

dataStack = False
if dataStack:
    masterData = np.zeros((ntraj, natoms, ntimePoints))
    coordinates = np.zeros((ntraj, natoms, 3))
    species = np.zeros((ntraj, natoms), dtype=np.int64)

    masterData[:ntraj55, :55, :] = masterData55
    masterData[ntraj55:ntraj55+ntraj146, :146, :] = masterData146
    masterData[ntraj55+ntraj146:, :, :] = masterData147
    coordinates[:ntraj55, :55, :] = coordinates55
    coordinates[ntraj55:ntraj55+ntraj146, :146, :] = coordinates146
    coordinates[ntraj55+ntraj146:, :, :] = coordinates147
    species[:ntraj55, :55] = np.ones(55)*Z
    species[ntraj55:ntraj55+ntraj146, :146] = np.ones(146)*Z
    species[ntraj55+ntraj146:, :] = np.ones(147)*Z

def make_master_data():
    masterData = np.zeros((ntraj, natoms, ntimePoints))
    coordinates = np.zeros((ntraj, natoms, 3))
    species = np.zeros((ntraj, natoms), dtype=np.int64)
    for idata in range(ntraj55):  # 3 NC types therefore
        mdata = idata * 3
        masterData[mdata, :55, :] = masterData55[idata,:,:]
        masterData[mdata+1, :146, :] = masterData146[idata,:,:]
        masterData[mdata+2, :, :] = masterData147[idata,:,:]
        
        coordinates[mdata, :55, :] = coordinates55
        coordinates[mdata+1, :146, :] = coordinates146
        coordinates[mdata+2, :, :] = coordinates147

        species[mdata, :55] = np.ones(55)*Z
        species[mdata+1, :146] = np.ones(146)*Z
        species[mdata+2, :] = np.ones(147)*Z

    return masterData, coordinates, species

#Draw random t points and not a subsequent intervals in a trajectory
def make_data():
    """
    Make data/batches based on the factor
    """
    #Draw random t-points for each trajectory
    if not dataStack: 
        masterData, coordinates, species = make_master_data()

    x_traj = np.zeros((nSys, natoms, nhist_here))
    coord = np.zeros((nSys, natoms, 3))
    spec = np.ones((nSys, natoms), dtype=np.int64)*Z
    for itraj in range(nSys):
        if TAG == 0:
            rand_index = np.random.choice(ntimePoints-nhist_here)
            x_traj[itraj, :, :] = masterData[itraj % ntraj,:, rand_index:rand_index+nhist_here]
        coord[itraj, :, :] = coordinates[itraj % ntraj, :, :]
        spec[itraj,:] = species[itraj % ntraj, :]
    z_tensor = torch.from_numpy(spec)
    r_tensor = torch.from_numpy(coord)
    
    if TAG == 0:
        traj_tensor = torch.from_numpy(x_traj)
        np.save(outFileDir+'traj.npy', x_traj)
    else:
        traj_tensor = torch.from_numpy(np.load(outFileDir+'traj.npy'))
    print('Made input data tensors', flush=True) #TODO
    
    #traj_tensor_1=traj_tensor[torch.randperm(nSys)]
    return z_tensor, r_tensor, traj_tensor

def load_HIPPYNN(trainModelDir):
    """
    Load the trained HIPPNN model
    """
    if TAG == 0:
        bundle = load_checkpoint(trainModelDir+'/experiment_structure.pt',
                             trainModelDir+'/best_checkpoint.pt',
                             restore_db=False, map_location = "cpu")#torch.device(device))
        model = bundle["training_modules"].model
        optimizer = torch.optim.Adam(model.parameters(),lr=init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patienceVal-1)
        epoch = 0
    else: 
        checkpoint = torch.load(outFileDir+'checkpoint.pth') 
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['lr_sched']
        epoch = checkpoint['epoch']
    
    print('Loaded the model', flush=True) #TODO
    v = optimizer.param_groups[0]["lr"]
    print("Initial earning rate:{:>10.5g}".format(v))
    return model, optimizer, scheduler, epoch

def get_train_val_data(splitpc):
    """
    Make random split of trajectories to train and valid
    """
    
    #Make current state from a random point in nhist=50
    val_size = int(splitpc*nSys)
    train_size = nSys - int(splitpc*nSys)
    print("Train size", train_size, "Validation size", val_size)
    train_subset, val_subset = torch.utils.data.random_split(
    traj_tensor, [train_size, val_size], generator=torch.Generator().manual_seed(143))
    train_traj = train_subset.dataset.data[train_subset.indices]
    val_traj = val_subset.dataset.data[val_subset.indices]
    
    #Current train data
    z_cur = z_tensor[train_subset.indices,:]
    r_cur = r_tensor[train_subset.indices,:,:]

    #Current validation data
    z_val = z_tensor[val_subset.indices,:]
    r_val = r_tensor[val_subset.indices,:,:]

    return train_traj, z_cur, r_cur, val_traj, z_val, r_val

def loss_recurr_infer(traj_input, r_input, z_input, pred_output, train=True):
    """
    Run the recurrent loop
    """
    loss_vals = []
    lossRMSE_track_ibatch = []
    lossMAE_track_ibatch = []
    lossQcons_track_ibatch = []
    endData = idataStop
    startData = endData - (nhist-1)
    q_input_cur = traj_input[:,:,startData:endData]
    
    for idata in range(endData, endData+n_recurr):
        outputs = pred_output(coordinates=r_input, dfeatures=q_input_cur, species=z_input)
        new_x = outputs['HDensityPred.atom_charges']; #print(idata, q_input_cur.shape)
        target_x = torch.unsqueeze(traj_input[:,:,idata], 2).to(device)
        error_RMSE = ((new_x-target_x)**2).mean().sqrt()
        error_MAE = (new_x-target_x).abs().mean()
        #error_L2 = 
        avgQ = new_x.sum(dim=0)
        error_Qcons = ((avgQ)*(avgQ)).mean().sqrt() #charge conservation
        if train:
            lossRMSE_track_ibatch.append(error_RMSE.item())
            lossMAE_track_ibatch.append(error_MAE.item())
            lossQcons_track_ibatch.append(error_Qcons.item())
        loss_vals.append(error_RMSE+error_MAE+0*error_Qcons)  #adding all the errors for training
        q_input_cur = torch.cat([q_input_cur[:,:,1:].to(device),new_x],dim=2) 
    
    return loss_vals, lossRMSE_track_ibatch, lossMAE_track_ibatch, lossQcons_track_ibatch

def get_batches_loss(traj_input, r_input, z_input, pred_output, train=True):
    """
    Run recurrence on all batches in the data
    """
    #Make batches
    nSample = traj_input.shape[0]
    if (nSample % batch_size) != 0:
        raise AssertionError()
        
    nBatches = nSample // batch_size
    batch_start = 0
    batch_end = batch_size
    batches_loss_values = []
    all_batchs_RMSE_track = np.zeros(n_recurr)
    all_batchs_MAE_track = np.zeros(n_recurr)
    all_batchs_Qcons_track = np.zeros(n_recurr)
    optimInterval = 1
    
    for ibatch in range(nBatches):
        if train: # and ibatch % optimInterval == 0:
            optimizer.zero_grad() #TODO
        
        #Make this batch inputs
        ibatch_traj_input = traj_input[batch_start:batch_end, :, :]
        ibatch_r_input = r_input[batch_start:batch_end, :, :]
        ibatch_z_input = z_input[batch_start:batch_end, :]
        
        #Do the recurrent training on this batch
        ibatch_loss_values, ibatch_RMSE_track, ibatch_MAE_track, ibatch_Qcons_track = loss_recurr_infer(ibatch_traj_input, 
                                                                                                           ibatch_r_input,
                                                                                                            ibatch_z_input,
                                                                                                              pred_output,
                                                                                                                train=True)
        loss_ibatch = sum(ibatch_loss_values)/nBatches #optimInterval #nBatches
        loss_ibatch.backward()
        batches_loss_values.append(loss_ibatch.item())
        batch_start = batch_end
        batch_end += batch_size
        all_batchs_RMSE_track += ibatch_RMSE_track
        all_batchs_MAE_track += ibatch_MAE_track
        del ibatch_loss_values
        if train: # and ibatch % optimInterval == 0:
            optimizer.step()  #TODO
        #Show progress
        if ibatch % 100 == 0:
            print(np.ceil(100*(ibatch/nBatches)), "%", "...", end='', flush=True)
    
    return batches_loss_values, all_batchs_RMSE_track, all_batchs_MAE_track, all_batchs_Qcons_track 
        
     
#Do the recurrent training
loss_history=[]
val_loss_history = []
lossRMSE_track = []
lossMAE_track = []
lossQcons_track = []

#Make the data tensors
z_tensor, r_tensor, traj_tensor = make_data()
train_traj, z_cur, r_cur, val_traj, z_val, r_val = get_train_val_data(splitpc)
#Load model
model, optimizer, scheduler, epoch = load_HIPPYNN(trainModelDir)
lr = optimizer.param_groups[0]['lr']
#Predictor
pred_output = Predictor.from_graph(model, return_device=device, model_device=device)
pred_output.requires_grad=True

bestModel = model
lowest_val_loss = 0
epochBest = epoch
doTrain = False
while  epoch < n_iterations: #r > threshold or epoch < n_iterations:
    
    #optimizer.zero_grad()
    
    #Random start time for this batch
    idataStop = torch.randint((nhist-1), nhist_here-n_recurr,())
    #print(idataStop, "picked at epoch", epoch)
    
    #Timer for recurrent training loop
    start = timeit.default_timer()
    print("Batch progress ....", end = " ", flush = True)
    
    #Run recurrent on all batches
    if epoch > 0 or TAG > 0:
        doTrain = True
    losses, i_RMSE_track, i_MAE_track, i_Qcons_track = get_batches_loss(train_traj, r_cur, z_cur, pred_output, train=doTrain)
    print("Done.")
    
    #Record for history
    loss_history.append(sum(losses))
    lossRMSE_track.append(i_RMSE_track)
    lossMAE_track.append(i_MAE_track)
    lossQcons_track.append(i_Qcons_track)
    
    #Update the model based on those gradients
    #optimizer.step()
    
    #Validate
    if epoch % 1 == 0.0:
        print("Validating ....", end = " ", flush=True)
        val_loss_values, _, _, _ = get_batches_loss(val_traj, r_val, z_val, pred_output, False)
        val_loss = sum(val_loss_values) 
        if epoch == 0 or val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            bestModel = model
            checkpoint = { 
            'epoch': epoch,
            'model': bestModel,
            'optimizer': optimizer,
            'lr_sched': scheduler}
            torch.save(checkpoint, outFileDir+'checkpoint.pth')
            epochBest = epoch
        val_loss_history.append(val_loss)
        print("Done.")
        print('Validation loss: {:0.4f}'.format(val_loss))
        print('Train loss: {:0.4f}'.format(sum(losses)))
        
        hippynn.tools.print_lr(optimizer)
        #Revert back to the model with lowest val err 
        if (epoch - epochBest) > patienceVal:
            print("Reverting back to the model with lowest val error", lowest_val_loss, " at epoch", epochBest, ".")  
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            old_lr = float(optimizer.param_groups[0]['lr'])
            #optimizer.param_groups['lr'] = old_lr*0.8  #Reduce by 20%
            hippynn.tools.print_lr(optimizer)
            epochBest = epoch

        scheduler.step(val_loss)
        #Print some progress
        print('Iter ', epoch, 'time (s): {:0.4f}'.format(timeit.default_timer() - start), end=" ", flush=True)
        print('======================')
        
        #Save losses for plotting
        np.save(outFileDir+"total_loss_history-"+str(TAG)+".npy", np.array(loss_history))
        np.save(outFileDir+"val_loss-"+str(TAG)+".npy", val_loss_history)
    
    lr = optimizer.param_groups[0]['lr']
    endTime = timeit.default_timer() 
    duration += endTime - start1
    start1 = endTime
    epoch += 1

print('Total Duration (hr): {:0.4f}'.format(duration/3600), flush=True)

#Save model and loss
lossRMSE_track = np.array(lossRMSE_track).reshape(epoch, n_recurr)
lossMAE_track = np.array(lossMAE_track).reshape(epoch, n_recurr)
lossCons_track = np.array(lossQcons_track).reshape(epoch, n_recurr)
np.save(outFileDir+"lossRMSE-"+str(TAG)+".npy", lossRMSE_track)
np.save(outFileDir+"lossMAE-"+str(TAG)+".npy", lossMAE_track)
np.save(outFileDir+"lossQcons-"+str(TAG)+".npy", lossQcons_track)
