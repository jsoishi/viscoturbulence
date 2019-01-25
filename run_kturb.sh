#!/usr/bin/bash
##SBATCH --ntasks-per-node=16
#SBATCH --partition=faculty
#SBATCH --time=5-0
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --distribution=cyclic:cyclic
#SBATCH --cpus-per-task=1

source /home/joishi/build/dedalus_intel_mpi/bin/activate

date
mpirun -np 8 python3 kturb.py run_H.cfg
date
