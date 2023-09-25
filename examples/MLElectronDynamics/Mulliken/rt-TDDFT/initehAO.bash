#!/bin/bash

for iEdir in {179..179}; do #1 2 3 4 5 6 9 40 72; do 
	export iEdir
	cd Edir-${iEdir}
	ln -sf ../gs.gpw ./
	ln -sf ../unocc.gpw ./
	ln -sf ../overlap.npy ./
	cp ../e-h-overTime-Umat.py ./
	cat > tmp-eh-${iEdir}.job << EOF
#!/bin/bash
### Begin BSUB Options
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH -p standard
#SBATCH -A w23_ml4chem
source ~/.bashrc
conda activate ch-GPAWTest

mpirun -n 1 python e-h-overTime-Umat.py
EOF
	sbatch tmp-eh-${iEdir}.job
	cd ../
done

