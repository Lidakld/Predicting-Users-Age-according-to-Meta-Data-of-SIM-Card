#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 18:18:43 2018

@author: lidakuang
"""
#%%
import numpy as np
import pandas as pd
#%%
loc='/Users/lidakuang/workspace/cs597_project/data/preprocessed_data.csv'

data = pd.read_csv(loc,index_col=0)
data.index = np.arange(0, len(data))
data = data.drop('X', axis = 1)

X = data.iloc[:,0:12].values
y = data['AGE'].values

#%%
from sklearn.cluster import MeanShift
clustering = MeanShift(bandwidth=200).fit(X)
clusters = np.nonzero(clustering.labels_)[0]