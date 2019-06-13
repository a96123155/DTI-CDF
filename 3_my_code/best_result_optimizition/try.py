
# coding: utf-8

# In[1]:

import sys
sys.path.append(r'E:\11_MFDF_code\3_my_code')
import os
os.chdir(r'E:\11_MFDF_code\1_original_data\NR')
from Graph_utils import *
from functions import *
from SNF import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_recall_curve
from sklearn.preprocessing import MaxAbsScaler
from  sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import gcforest
from gcforest.gcforest import GCForest
import xgboost
from functools import reduce
import glob
import matplotlib.pyplot as plt
from scipy import interp
import csv
np.set_printoptions(threshold=np.inf)

with open(r'E:\11_MFDF_code\9_y_pred\6.1_training_NR_MFDF_rf3.csv', 'wb') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerow(['Each_row: 1.mode; 2.layer; 3.aupr; 4.auc; 5.y'])
    wr.writerow(mode_list)
    wr.writerow(layer)
    wr.writerow(aupr_list)
    wr.writerow(auc_list)
    wr.writerow(test_true_predict_compare_10cv_seeds_modes)
        
