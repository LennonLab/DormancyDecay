#!/bin/bash
#PBS -k o
#PBS -l nodes=1:ppn=10,vmem=150gb,walltime=72:00:00
#PBS -M mmuscare@indiana.edu,lennonj@indiana.edu,kjlocey@indiana.edu
#PBS -m abe
#PBS -j oe
module load gcc
module load mothur/1.32.1
cd /N/dc2/projects/Lennon_Sequences/2015_INPonds
mothur INPonds_A.batch
