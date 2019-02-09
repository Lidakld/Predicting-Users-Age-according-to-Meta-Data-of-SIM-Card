#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:47:01 2018

@author: lida kuang
"""
#%%
import numpy as np
import pandas as pd
#%%
loc = '/Users/matthewxfz/workspace/kuangkuang/cs597/data/preprocessed_data.csv'
data = pd.read_csv(loc, index_col=0)
#%%
data = data.drop('X', axis = 1)
#%%
age_type = []
AGE = data['AGE']
for i in range (1,len(AGE)+1):
    if AGE[i] > 10 and AGE[i] <= 15:
        age_type.append(0)
    elif AGE[i] > 15 and AGE[i] <= 20:
        age_type.append(1)
    elif AGE[i] > 20 and AGE[i] <= 25:
        age_type.append(2)
    elif AGE[i] > 25 and AGE[i] <= 30:
        age_type.append(3)
    elif AGE[i] > 30 and AGE[i] <= 35:
        age_type.append(4)
    elif AGE[i] > 35 and AGE[i] <= 40:
        age_type.append(5)
    elif AGE[i] > 40 and AGE[i] <= 45:
        age_type.append(6)
    elif AGE[i] > 45 and AGE[i] <= 50:
        age_type.append(7)
    elif AGE[i] > 50 and AGE[i] <= 55:
        age_type.append(8)
    else:
        age_type.append(9)
age_type = np.asarray(age_type)

#%%
data_binned = data.drop('AGE', axis = 1)
data_binned['AGE_TYPE'] = age_type
