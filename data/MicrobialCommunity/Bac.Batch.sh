#!/bin/bash
#PBS -k o
#PBS -l nodes=1:ppn=32,vmem=100gb,walltime=72:00:00
#PBS -M srbray@indiana.edu
#PBS -m abe
#PBS -j oe
module load gcc
module load mothur/1.31.2
cd /N/dc2/projects/Lennon_Sequences/INPonds2014/
mothur INPonds.Batch
