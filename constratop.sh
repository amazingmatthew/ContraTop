#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=2

module load compilers/gcc/8.2.0
module load apps/binapps/anaconda3/2022.10
module load libs/cuda
module load tools/bintools/ninja/1.10.0

# This is the executable which is run by the batch system to do our calculation
python3 topicmodeling.py