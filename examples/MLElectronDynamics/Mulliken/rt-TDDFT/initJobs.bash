#!/bin/bash

for iEdir in {231..245}; do
	export iEdir
	mkdir Edir-${iEdir}
	cd Edir-${iEdir}
	ln -sf ../gs.gpw  
	python ../make_rand_input_file.py 
	cat > tmp-${iEdir}.job << EOF
#!/bin/bash
### Begin BSUB Options
#SBATCH -t 16:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH -p standard
#SBATCH -A w21_mlmat

source ~/.bashrc
conda activate ch-GPAWTest

mpirun -n 32 python td-time-Emag.py
EOF
	sbatch tmp-${iEdir}.job
	cd ../
done

