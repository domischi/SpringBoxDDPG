#!/bin/bash
#SBATCH --job-name="Springbox Learning"
#SBATCH --ntasks=8
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dominiks@caltech.edu
#SBATCH --output=out
#SBATCH --error=out
#======START===============================
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST"
echo "A total of $SLURM_NTASKS tasks is used"
CMD="python3 PBT.py"
echo $CMD
$CMD
#======END================================= 
