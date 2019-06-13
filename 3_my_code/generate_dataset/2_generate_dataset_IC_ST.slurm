#!/bin/bash
#SBATCH -J gener_IC_ST
#SBATCH -p cpu
#SBATCH --mail-type=end
#SBATCH --mail-user=a96123155@sjtu.edu.cn
#SBATCH -o /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/generate_dataset/log_generate_dataset_IC_ST.out
#SBATCH -e /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/generate_dataset/log_generate_dataset_IC_ST.err
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --qos=qoslong

module load miniconda3
source activate py27
cd /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_my_code/generate_dataset
python -u 2_generate_dataset_IC_ST.py > /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/generate_dataset/generate_dataset_IC_ST.out