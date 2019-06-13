#!/bin/bash
#SBATCH -J E_SD_12_FP2_PSSM_env_new
#SBATCH -p cpu
#SBATCH --mail-type=end
#SBATCH --mail-user=a96123155@sjtu.edu.cn
#SBATCH -o /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/training_FP2_PSSM/log_E_SD_12_FP2_PSSM_env_new.out
#SBATCH -e /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/training_FP2_PSSM/log_E_SD_12_FP2_PSSM_env_new.err
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --qos=qoslong

module load miniconda3
source activate py2_errauc
cd /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_my_code/training_FP2_PSSM
python -u 2_training_E_SD_PSSM_FP2_12.py > /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/training_FP2_PSSM/E_SD_12_FP2_PSSM_env_new.out