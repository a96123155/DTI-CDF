#!/bin/bash
#SBATCH -J IC_7_DDR_env_new
#SBATCH -p cpu
#SBATCH --mail-type=end
#SBATCH --mail-user=a96123155@sjtu.edu.cn
#SBATCH -o /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/training_DDR/log_IC_7_DDR_env_new.out
#SBATCH -e /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/training_DDR/log_IC_7_DDR_env_new.err
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --qos=qoslong

module load miniconda3
source activate py2_errauc
cd /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_my_code/training_DDR
python -u 2_training_IC_DDR_12.py > /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/training_DDR/IC_7_DDR_env_new.out