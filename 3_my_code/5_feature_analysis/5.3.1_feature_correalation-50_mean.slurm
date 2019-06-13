#!/bin/bash
#SBATCH -J 5.3.1_50_mean_feature_correalation
#SBATCH -p cpu
#SBATCH --mail-type=end
#SBATCH --mail-user=a96123155@sjtu.edu.cn
#SBATCH -o /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_my_code/5_feature_analysis/log_cross_correlation.out
#SBATCH -e /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_my_code/5_feature_analysis/log_cross_correlation.err
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --qos=qoslong

module load miniconda3
source activate py27
cd /lustre/home/acct-clsdqw/clsdqw-tahir/dqw_cyy/11_MFDF_code/3_my_code/5_feature_analysis
python -u 5.3.1_feature_correalation-50_mean.py > 5.3.1_feature_correalation-50_mean.out