import sys
if len(sys.argv) < 7:
    print('python trainHIPNN.py <feature size>, <data_name prefix>, <dataset_path>, <output dir name>, <TAG 0 or 1> <gpu # >')
    exit();

import os
import matplotlib
import hippynn
import numpy as np
import torch
dtype=torch.float64
torch.set_default_dtype(dtype)
if torch.cuda.is_available():
    device = 'cuda:'+cudaDevice
    torch.cuda.set_device(int(cudaDevice))
else:
    device = 'cpu'
seed = 137800
torch.manual_seed(seed)
#torch.random.seed()
hippynn.custom_kernels.set_custom_kernels (False)
featureSize = int(sys.argv[1]) # the number of histories (nhist-1)
dataset_name = sys.argv[2] #'Ag-4-100x-'    # Prefix for arrays in the data folder
dataset_path = sys.argv[3] #"../data"
netname = sys.argv[4] #nhis4-1InteractionLayer'
TAG = int(sys.argv[5])  # False (0): first run, True(n): continue
cudaDevice = sys.argv[6]
Z = 47 #Only silver atoms so far
dirname = netname
if not os.path.exists(dirname):
    os.mkdir(dirname)
else:
    pass

# Log the output of python to `training_log.txt`
with hippynn.tools.log_terminal("training_log_tag_%d.txt" % TAG,'wt'): #and torch.autograd.set_detect_anomaly(True):
    #Model parameters
    dist_hard_max = 10
    possible_species=[0, Z] # actual species in the dataset
    feature_size=featureSize
    network_params = {
        "possible_species": [1 for i in range(feature_size+1)],   # Something the length of your inputs plus 1
        'n_features': 30,                     # Number of neurons at each layer
        "n_sensitivities": 10,                # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 1.5,                 #
        "dist_soft_max": 5.,
        "dist_hard_max": dist_hard_max,
        "n_interaction_layers": 3,            # Number of interaction blocks
        "n_atom_layers": 3,                   # Number of atom layers in an interaction block
    }
    
    from hippynn.graphs import inputs, networks, targets, physics
    from hippynn.graphs.nodes import indexers, pairs
    from hippynn.graphs.indextypes import IdxType
    #Model inputs
    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    other_features = inputs.InputNode(db_name='dfeatures')
    other_features._index_state = IdxType.MolAtom
    enc, padidxer = indexers.acquire_encoding_padding(species, species_set=possible_species)
    pairfinder = pairs.OpenPairIndexer('PairFinder', 
                                       (positions,species),
                                       dist_hard_max=dist_hard_max)
    #Model computations
    network = networks.Hipnn("HIPNN", 
                            (other_features,pairfinder),
                             periodic=True,
                             module_kwargs = network_params)
    pred_el_pop_pred = targets.HChargeNode("HDensityPred",network)
    pred_el_pop = pred_el_pop_pred.main_output
    pred_el_pop.db_name="dtargets"
    
    #Dipole node
    pred_dipole = physics.DipoleNode("dipoleMoment", (pred_el_pop_pred,positions))
    pred_dipole.db_name="dipoleMoment"
    
    #Apply charge conservation
    charge_sums_predicted = physics.AtomToMolSummer("AvgQ2", pred_el_pop).pred 
    
    #Define loss quantities
    from hippynn.graphs import loss
    rmse_el_pop = loss.MSELoss.of_node(pred_el_pop) ** (1 / 2)  
    mae_el_pop = loss.MAELoss.of_node(pred_el_pop)
    rsq_el_pop = loss.Rsq.of_node(pred_el_pop)
    loss_error = (rmse_el_pop + mae_el_pop)
    l2_reg = loss.l2reg(network)
    loss_regularization = 1e-6 * loss.Mean(l2_reg)    # L2 regularization 
    charge_conserv_loss = loss.Mean(charge_sums_predicted*charge_sums_predicted)
    train_loss = loss_error + loss_regularization + charge_conserv_loss
 
    # Validation losses are what we check on the data between epochs -- we can only train to
    # a single loss, but we can check other metrics too to better understand how the model is training.
    # There will also be plots of these things over time when training completes.
    validation_losses = {
        "Q-RMSE"      : rmse_el_pop,
        "Q-MAE"       : mae_el_pop,
        "Q-RSQ"       : rsq_el_pop,
        "L2Reg"       : l2_reg,
        "Loss-Err"    : loss_error,
        "Loss-Reg"    : loss_regularization,
        "Loss"        : train_loss,
        "Charge-conserve-loss" : charge_conserv_loss
    }
    early_stopping_key = "Loss-Err"
    
    #Make correlation plots for visualization
    from hippynn import plotting
    plot_maker = plotting.PlotMaker(
        plotting.Hist2D.compare(pred_el_pop, saved="el-pop.pdf"),
        plot_every=200)

    if TAG==0: # TAG for restarting the calculations
        from hippynn.experiment.assembly import assemble_for_training
        training_modules, db_info = \
            assemble_for_training(train_loss,validation_losses,plot_maker=plot_maker)
        training_modules[0].print_structure()
        #Read the database
        database_params = {
            'name': dataset_name,           # Prefix for arrays in folder
            'directory': dataset_path,
            'quiet': False,                 # Quiet==True: suppress info about loading database
            'seed': seed,                   # Random seed for data splitting
            'test_size': 0.2,               # Fraction of data used for testing
            'valid_size':0.2,               # Fraction of data used for validation
               **db_info                    # Adds the inputs and targets names from the model as things to load
        }

        from hippynn.databases import DirectoryDatabase
        database = DirectoryDatabase(**database_params)
        
        #Controller and optimizer
        init_lr = 1e-3
        optimizer = torch.optim.Adam(training_modules.model.parameters(),lr=init_lr)
        from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController
        scheduler =  RaiseBatchSizeOnPlateau(optimizer=optimizer,
                                            patience=50,
                                            factor=0.5,
                                            max_batch_size=80)

        controller = PatienceController(optimizer=optimizer,
                                        scheduler=scheduler,
                                        batch_size=64,
                                        eval_batch_size=64, 
                                        max_epochs=10000,
                                        termination_patience=500,
                                        fraction_train_eval=1.,
                                        stopping_key=early_stopping_key,
                                        )


        experiment_params = hippynn.experiment.SetupParams(controller = controller, device=device)
        # Parameters describing the training procedure.
        from hippynn.experiment import setup_training
        training_modules, controller, metric_tracker = setup_training(training_modules=training_modules,
                                                        setup_params=experiment_params)
    if TAG>0:
        from hippynn.experiment.serialization import load_checkpoint_from_cwd, load_checkpoint
        from hippynn.experiment import train_model
        #load best model
        structure = load_checkpoint_from_cwd()
        #load last model
        #structure = load_checkpoint("experiment_structure.pt", "last_checkpoint.pt")
        training_modules = structure["training_modules"]
        database = structure["database"]
        controller = structure["controller"]
        metric_tracker = structure["metric_tracker"]
    
    #Train the model
    from hippynn.experiment import train_model
    store_all_better=False
    store_best=True
    train_model(training_modules=training_modules,
                database=database,
                controller=controller,
                metric_tracker=metric_tracker,
                callbacks=None,batch_callbacks=None,
                store_all_better=store_all_better,
                store_best=store_best) 
