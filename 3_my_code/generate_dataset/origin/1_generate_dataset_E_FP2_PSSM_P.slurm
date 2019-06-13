#!/bin/bash
#SBATCH -J 1_E_SP_FP2_PSSM
#SBATCH -p cpu
#SBATCH --mail-type=end
#SBATCH --mail-user=a96123155@sjtu.edu.cn
#SBATCH -o /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/generate_dataset/log_1_generate_dataset_E_FP2_PSSM_SP.out
#SBATCH -e /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/generate_dataset/log_1_generate_dataset_E_FP2_PSSM_SP.err
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --qos=qoslong

module load miniconda3
source activate py27
cd /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_my_code/generate_dataset
python -u 1_generate_dataset_E_FP2_PSSM_P.py > /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/generate_dataset/1_generate_dataset_E_FP2_PSSM_SP.out