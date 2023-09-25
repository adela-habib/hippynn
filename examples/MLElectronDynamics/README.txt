#in rt-TDDFT folder for data generation
1. Install GPAW and use the GLLSB-sc pseudopotential file.
2. Run ground SCF calculation with specified lcao mode, number of Kohn-Sham bands. (use mpi processes in gs.py)
3. Run real-time TDDFT with LR and collect wavefunctions. Check td-time-Emag.py. To make different trajectories change Emag, Ehat within td-time-Emag.py.
4. To restart and continue dynamics, check tdc-time-Emag.py.
5. To get Mullikan populations over time and per atom, run get_el_pop_time.py. This needs Mullikan population (GAP_t0.npy)
before pulse perturbation and the overlap matrix as well as number of time steps.
6. Once, we have this data (MullikenElectronPop.npy) for each trajectory, we can start training ML models. For large systems (e.g., Ag561) this data collection takes long. It can be done in batches and then collected using collect-MullPop.py.

#in single-step-training folder and in data folder
1. First in the data folder, we make training features and target data by chopping the trajectories in intervals. 
2. Make_ml_train_interval.py reads all the data npy files with suffix "EmagEdirTest" for train and "TestPaper" for held-out test. The "EmagEdirTest" data can be provided upon demand. The files are bigger than 100MB limit for github. 
3. In single-step-training, trainHIPNN.py trains a charge model in the model directory

#in multi-step-training folder and in data folder
1. train-recurr.py takes some inputs as mentioned and multi-step wise trains to future data
2. The nRecurr input determines how far in future should we train to
3. It needs reading an initially single-step trained model as mentioned above

#in multi-step-prediction folder for long time charge trajectory predictions
1. MLCharge.py runs the model to make long time predictions
2. MLDipole.py reads the ML predicted charges and calculates dynamic dipoles. Check out how to read atoms. 
