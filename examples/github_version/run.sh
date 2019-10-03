#!/bin/bash

nvcc -arch=sm_52 --compiler-options="-O2 -fopenmp" -o myVegas myVegasMain.cu

for dim in {6..12};
do
	for pts in {8..16};
	do
		./myVegas -d ${dim} -n ${pts} -i 10 -a 10
	done
done
