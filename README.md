# DTI-CDF
Drug-target interactions (DTIs) play a crucial role in target-based drug discovery and development. Computational prediction of DTIs has become a popular supplementary strategy to conduct the experimental methods, which are both time as well as resource consuming, for identification of DTIs. However, the performances of the current DTIs prediction approaches suffer from a problem of low precision and high false positive rate. In this study, we aim to develop a novel DTIs prediction method, named DTI-CDF, for improving the prediction performance based on a cascade deep forest model with multiple similarity-based features extracted from the heterogeneous graph. In the experiments, we build five replicates of 10 fold cross-validations under three different experimental settings of data sets, namely, corresponding DTIs values of certain drugs (SD), targets (ST), or drug-target pairs (SP) in the training sets are missed but existed in the test sets. The experimental results demonstrate that our proposed approach DTI-CDF achieves significantly higher performance than of the state-of-the-art methods. And there are 1352 predicted new DTIs are proved correct by KEGG and DrugBank databases.

#-----------------------------------------------------------------------------------------------------------------------------
Requirements and Run Nuclear Receptors data set (as an example):

1. Python2.7 Environment

2. pip install -r DTI-CDF_requirements.txt

3. install DTI-CDF/gcforest: please move this fold to your python environment, such as dir = '\Anaconda3\envs\ipykernel_py2\Lib\site-packages'.

4. download fold '.\DTI-CDF\2_Example_NR'

5. cd .\DTI-CDF\2_Example_NR

6. python -u Example_NR.py > Example_NR.out

Then, you could see the results in 'Example_NR.out' file.
