#!/bin/bash

for iEdir in {7..20}; do
	export iEdir
	#mkdir Edir-${iEdir}
	cd Edir-${iEdir}
	cp ../get_el_pop_time.py ./
	cp ../launchGetMull.sh ./
       	cd ../
done
