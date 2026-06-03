#!/bin/bash -l
#SBATCH --job-name=svg_to_numpy					# Give your job a name, so you can recognize it in the queue overview
#SBATCH --nodes=1						# Define, how many nodes you need. Here, we ask for 1 node..
#SBATCH --cpus-per-task=32
#SBATCH --partition=quickgpuq					# other options are: {bigmemq | defq | longrunq | tinyq]  gpuq
#SBATCH --gres=gpu:1						# Set to ‘gpu:2’ for 2 GPU’s
#SBATCH --mem=128g						# Set RAM memory limit to 64GB - choose up to the max on that node
#SBATCH --mail-type=END,FAIL					# Turn on mail notification. There are many possible self-explaining values: [NONE, BEGIN, END, FAIL, ALL]
#SBATCH --time=0-02:00:00					# Time limit hrs:min:sec (here we set a 2 hour limit)
#SBATCH --mail-user=agonzalez9adf2@houstonmethodist.org		# add your email address
#SBATCH --output=svg_to_numpy_%j.log  				# Standard output and error log
set -e

pwd; hostname; date
#
## Load modules needed for this job
#
module load mamba

conda activate pdf_ecg
#
echo "Starting the script"
#
python /condo/alkindilab/hmaiaxg21/PDF_to_numpy_pipeline/1_pdf_svg_df_extract_antonio_Jun2.py --end_index 1000


#
date
#
# In addition to the copied files, you will also find a file called
# slurm-1234.out in the submit directory. This file will contain all output that
# was produced during runtime, i.e. stdout and stderr.
exit 0
