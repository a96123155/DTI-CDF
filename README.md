# DTI-CDF
## An `instruction` file for DTI-CDF for DTIs prediction.

### Introduction
Drug-target interactions (DTIs) play a crucial role in target-based drug discovery and development. Computational prediction of DTIs has become a popular supplementary strategy to conduct the experimental methods, which are both time as well as resource consuming, for identification of DTIs. However, the performances of the current DTIs prediction approaches suffer from a problem of low precision and high false positive rate. In this study, we aim to develop a novel DTIs prediction method, named DTI-CDF, for improving the prediction performance based on a cascade deep forest model with multiple similarity-based features extracted from the heterogeneous graph. In the experiments, we build five replicates of 10-fold cross-validations under three different experimental settings of data sets, namely, corresponding DTIs values of certain drugs (SD), targets (ST), or drug-target pairs (SP) in the training sets are missed but existed in the test sets. The experimental results demonstrate that our proposed approach DTI-CDF achieves significantly higher performance than of the state-of-the-art methods. And there are 1352 predicted new DTIs are proved correct by KEGG and DrugBank databases.


### Requirements
This method developed with Python 2.7, please make sure all the dependencies are installed, which is specified in DTI-CDF_requirements.txt.


### Reference
DTI-CDF a cascade deep forest model towards the prediction of drug-target interactions based on hybrid features.


### Run NR data set (as a demo)
1. Download fold “.\DTI-CDF\2_Example_NR”.

2. In the “.\DTI-CDF\2_Example_NR” path, run the Example_NR.py file, as follows:  
   Open CMD and input:  
          `cd .\DTI-CDF\2_Example_NR`  
          `python -u Example_NR.py > Example_NR.out`


Please see “Example_NR.out” file for the results/outputs which contains the results of performance metrics, time required for the program to run and the new DTIs predicted by this method.  
If you want to try other data sets, just follow this demo, and the codes and data have been supported in fold “1_all_code” and “1_original_data”, respectively.

### Package dependencies

The package is developed in python 2.7, higher version of python is not suggested for the current version.  
Run the following command to install dependencies before running the code: pip install -r DTI-CDF_requirements.txt.  
If something wrong when you run the code, you could reinstall gcforest as follow: move fold "DTI-CDF/gcforest" to your python environment, such as dir = '\Anaconda3\envs\ipykernel_py2\Lib\site-packages'.

### Others
Please read reference and py file for a detailed walk-through.

### Thanks
