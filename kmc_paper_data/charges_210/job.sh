#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --output=myjob.output
#SBATCH --error=myjob.error
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --account=free
#SBATCH --partition=batch
#SBATCH --time=06:00:00
#SBATCH --mail-user=jmd80@bath.ac.uk

snakemake --cores 16
