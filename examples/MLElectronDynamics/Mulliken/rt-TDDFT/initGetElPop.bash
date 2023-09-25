#!/bin/bash

for iEdir in {231..245}; do #103; do # 3 4 5 6 9 40 72 79; do 
	export iEdir
	#mkdir Edir-${iEdir}
	cd Edir-${iEdir}
	cp ../get_el_pop_time.py ./ #get_spdOrb_eh_pop_time.py ./ #get_eh_pop_time.py  ./
	cat > tmp-ml-${iEdir}.job << EOF
#!/bin/bash
### Begin BSUB Options
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -p standard
#SBATCH -A w21_mlmat
###cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
conda activate ch-GPAWTest

mpirun -n 1 python get_el_pop_time.py #get_spdOrb_eh_pop_time.py #get_eh_pop_time.py
EOF
	sbatch tmp-ml-${iEdir}.job
	cd ../
done

