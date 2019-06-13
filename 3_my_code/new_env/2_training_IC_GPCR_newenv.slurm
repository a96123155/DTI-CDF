#!/bin/bash
#SBATCH -J IC_GPCR_newenv
#SBATCH -p cpu
#SBATCH --mail-type=end
#SBATCH --mail-user=a96123155@sjtu.edu.cn
#SBATCH -o /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/log_2_training_IC_GPCR_newenv.out
#SBATCH -e /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/log_2_training_IC_GPCR_newenv.err
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --qos=qoslong

module load miniconda3
source activate py2_errauc
cd /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_my_code/
python -u 2_training_IC_GPCR_newenv.py > /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_log_file/2_training_IC_GPCR_newenv.out