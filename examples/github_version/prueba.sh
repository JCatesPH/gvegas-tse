#!/bin/bash
for dim in {2..8}
do
for cores in 1 2 4 8 12
do
OMP_NUM_THREADS=${cores} perf stat -r40 -o ./datos/runtime/gVegas${cores}/d${dim}n0.dat ./gVegas -d ${dim} -n 0 -a 1 -i 10 > /dev/null
done
done
