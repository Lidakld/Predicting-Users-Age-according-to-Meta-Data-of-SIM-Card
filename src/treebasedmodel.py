#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:38:31 2018

@author: lidakuang
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import KFold
import xgboost as xgb
#%%
path='/Users/lidakuang/workspace/cs597_project/data/my_data.csv'

data = pd.read_csv(path,index_col=0)
data.index = np.arange(0, len(data))
X = data.drop('AGE',axis = 1)
Y = data['AGE']

#%%
#normalization
X_norm = (X-X.mean())/X.std()
#fill nas with mean
X_non_nan = X_norm.fillna(X_norm.mean())
data_non_nan = X_non_nan
data_non_nan['AGE']=Y
#delete outliers
outlier_idx = np.where(np.abs(X_non_nan-X_non_nan.mean(axis = 0)) > (3*X_non_nan.std()))[0]
outlier_idx = np.unique(outlier_idx)
data_non_outlier = data_non_nan.drop(data_non_nan.index[outlier_idx])
data_non_outlier = data_non_outlier.drop(['INSTAGRAM_MB','P4_INSTAGRAM_MB',
                                          'PERC_INSTAGRAM_MB_IN_P3','DEVICE_CAPABILITY_2G'],axis = 1)

data_new = data_non_outlier
X_new = data_new.drop('AGE',axis = 1)
Y_new = data_new['AGE']

f, ax = plt.subplots(figsize=(30, 30))
corr = data_new.corr()
sns.heatmap(corr, annot=True,ax = ax)

X_new.boxplot(rot = 90, figsize = (10,10))
#%%
#Random Forrest
X = X_new.values
y = Y_new.values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#%%
#RM---Training
kf = KFold(n_splits=5)
rmse_avg = 0
rmse_app_avg = 0
for train_index, test_index in kf.split(X_train):
    X_tr, X_te = X_train[train_index], X_train[test_index]
    y_tr, y_te = y_train[train_index], y_train[test_index]
    
    clf = RandomForestClassifier(n_estimators=200,oob_score = True, max_features=20,random_state=0)
    clf.fit(X_tr,y_tr)

    y_hat = clf.predict(X_te)
    
    sns.distplot(y_hat,kde=False,color = 'skyblue')
    sns.distplot(y_te,kde=False, color = 'y')
    plt.show()
    
    rmse = np.sqrt(mean_squared_error(y_te, y_hat))
    rmse_avg = rmse_avg + rmse
    print('rmse: %5.3f'% rmse)
    
    y_hat_appro = np.zeros(len(y_hat))
    for i in range(len(y_hat)):  
        if(np.abs(y_hat[i]-y_te[i])<=5):
            y_hat_appro[i]=y_te[i]
        else:
            y_hat_appro[i] = y_hat[i]
            
    rmse_appo = np.sqrt(mean_squared_error(y_te, y_hat_appro))
    rmse_app_avg = rmse_app_avg + rmse_appo
    print('rmse_appo: %5.3f'% rmse_appo)
    
print('rmse_avg: %5.3f'% (rmse_avg/5))
print('rmse_app_avg: %5.3f'% (rmse_app_avg/5))
#%%
#RM--Testing
y_hat = clf.predict(X_test)
sns.distplot(y_hat,kde=False,color = 'skyblue')
sns.distplot(y_test,kde=False, color = 'y')
plt.show()

rmse = np.sqrt(mean_squared_error(y_test, y_hat))
print('rmse: %5.3f'% rmse)

y_hat_appro = np.zeros(len(y_hat))
for i in range(len(y_hat)):  
    if(np.abs(y_hat[i]-y_test[i])<=5):
        y_hat_appro[i]=y_test[i]
    else:
        y_hat_appro[i] = y_hat[i]
        
rmse_appo = np.sqrt(mean_squared_error(y_test, y_hat_appro))
print('rmse_appo: %5.3f'% rmse_appo)
    
#%%
data_dmatrix = xgb.DMatrix(data = X, label = y)
X_train_xbg, X_test_xbg, y_train_xbg, y_test_xbg = train_test_split(X,y,test_size=0.2,random_state=0)
#%%
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.6, learning_rate = 0.1,
                max_depth = 3, alpha = 15, n_estimators = 100)
#%%
xbg_rmse_avg = 0
xbg_rmse_app_avg = 0

for train_index, test_index in kf.split(X_train_xbg):
    X_tr, X_te = X_train[train_index], X_train[test_index]
    y_tr, y_te = y_train[train_index], y_train[test_index]
    
    xg_reg.fit(X_tr,y_tr)

    y_hat = xg_reg.predict(X_te)
    
    sns.distplot(y_hat,kde=False,color = 'skyblue')
    sns.distplot(y_te,kde=False, color = 'y')
    plt.show()
    
    rmse = np.sqrt(mean_squared_error(y_te, y_hat))
    xbg_rmse_avg = xbg_rmse_avg + rmse
    print('rmse: %5.3f'% rmse)
    
    y_hat_appro = np.zeros(len(y_hat))
    for i in range(len(y_hat)):  
        if(np.abs(y_hat[i]-y_te[i])<=5):
            y_hat_appro[i]=y_te[i]
        else:
            y_hat_appro[i] = y_hat[i]
            
    rmse_appo = np.sqrt(mean_squared_error(y_te, y_hat_appro))
    xbg_rmse_app_avg = xbg_rmse_app_avg + rmse_appo
    print('rmse_appo: %5.3f'% rmse_appo)
    
print('rmse_avg: %5.3f'% (xbg_rmse_avg/5))
print('rmse_app_avg: %5.3f'% (xbg_rmse_app_avg/5))

#rmse = np.sqrt(mean_squared_error(y_test, y_hat_xg))

#%%
y_hat_xbg = xg_reg.predict(X_test_xbg)
sns.distplot(y_hat_xbg,kde=False,color = 'skyblue')
sns.distplot(y_test_xbg,kde=False, color = 'y')
plt.show()

rmse_xbg = np.sqrt(mean_squared_error(y_test_xbg, y_hat_xbg))
print('rmse: %5.3f'% rmse_xbg)

y_hat_appro_xbg = np.zeros(len(y_hat_xbg))
for i in range(len(y_hat_xbg)):  
    if(np.abs(y_hat_xbg[i]-y_test_xbg[i])<=5):
        y_hat_appro_xbg[i]=y_test_xbg[i]
    else:
        y_hat_appro_xbg[i] = y_hat_xbg[i]
        
rmse_appo_xbg = np.sqrt(mean_squared_error(y_test_xbg, y_hat_appro_xbg))
print('rmse_appo: %5.3f'% rmse_appo_xbg)

#%%
loc='/Users/matthewxfz/workspace/kuangkuang/Predicting-User-Age-by-Mobile-Phone-Meta-Data/cs597_project/data/preprocessed_data.csv'


data2 = pd.read_csv(loc,index_col=0)
data2.index = np.arange(0, len(data2))
data2 = data2.drop('X', axis = 1)

X2 = data2.drop('AGE',axis = 1)
Y2 = data2['AGE']
#%%
X2.boxplot(rot = 90,figsize = (10,8))
#%%
X2_norm = (X2-X2.mean())/X2.std()
X2_norm['AGE']=Y2
data2_norm = X2_norm
out2_idx = np.where(np.abs(X2_norm-X2_norm.mean(axis = 0)) > (2*X2_norm.std()))[0]
out2_idx = np.unique(out2_idx)
data2_non_outlier = data2_norm.drop(data2_norm.index[out2_idx])
#%%
data2_new = data2_non_outlier
X2_new = data2_new.drop('AGE',axis = 1)
Y2_new = data2_new['AGE']
#%%
X2_new.boxplot(rot = 90, figsize = (10,8))
#%%
#RM--trainning
X2 = X2_new.values
Y2 = Y2_new.values
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,Y2,test_size=0.2,random_state=0)

rmse2_avg = 0
rmse2_app_avg = 0
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X2_train):
    X_tr, X_te = X2_train[train_index], X2_train[test_index]
    y_tr, y_te = y2_train[train_index], y2_train[test_index]
    
    clf = RandomForestClassifier(n_estimators=200,oob_score = True, max_features=6,random_state=0)
    clf.fit(X_tr,y_tr)
    oob = clf.oob_score_
    print('oob: %5.3f' % oob)
    y_hat = clf.predict(X_te)
    
    sns.distplot(y_hat,kde=False,color = 'skyblue')
    sns.distplot(y_te,kde=False, color = 'y')
    plt.show()
    
    rmse = np.sqrt(mean_squared_error(y_te, y_hat))
    rmse2_avg = rmse2_avg + rmse
    print('rmse: %5.3f'% rmse)
    
    y_hat_appro = np.zeros(len(y_hat))
    for i in range(len(y_hat)):  
        if(np.abs(y_hat[i]-y_te[i])<=5):
            y_hat_appro[i]=y_te[i]
        else:
            y_hat_appro[i] = y_hat[i]
            
    rmse_appo = np.sqrt(mean_squared_error(y_te, y_hat_appro))
    rmse2_app_avg = rmse2_app_avg + rmse_appo
    print('rmse_appo: %5.3f'% rmse_appo)
    
print('rmse2_avg: %5.3f'% (rmse2_avg/5))
print('rmse2_app_avg: %5.3f'% (rmse2_app_avg/5))
#%%
#RM--testing
y2_hat = clf.predict(X2_test)
sns.distplot(y2_hat,kde=False,color = 'skyblue')
sns.distplot(y2_test,kde=False, color = 'y')
plt.show()

rmse2 = np.sqrt(mean_squared_error(y2_test, y2_hat))
print('rmse2: %5.3f'% rmse2)

y2_hat_appro = np.zeros(len(y2_hat))
for i in range(len(y2_hat)):  
    if(np.abs(y2_hat[i]-y2_test[i])<=5):
        y2_hat_appro[i]=y2_test[i]
    else:
        y2_hat_appro[i] = y2_hat[i]
        
rmse2_appo = np.sqrt(mean_squared_error(y2_test, y2_hat_appro))
print('rmse2_appo: %5.3f'% rmse2_appo)
#%%
#xbg--training
data_dmatrix2 = xgb.DMatrix(data = X2, label = Y2)
X2_train_xbg, X2_test_xbg, y2_train_xbg, y2_test_xbg = train_test_split(X2,Y2,test_size=0.2,random_state=0)
#%%
xg2_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.6, learning_rate = 0.1,
                max_depth = 3, alpha = 15, n_estimators = 100)
#%%

#%%
xbg2_rmse_avg = 0
xbg2_acc_avg = 0
xbg2_acc_appo = 0

for train_index, test_index in kf.split(X2_train_xbg):
    X_tr, X_te = X2_train[train_index], X2_train[test_index]
    y_tr, y_te = y2_train[train_index], y2_train[test_index]
    
    xg2_reg.fit(X_tr,y_tr)
    y_hat = xg2_reg.predict(X_te)
    
    sns.distplot(y_hat,kde=False,color = 'skyblue')
    sns.distplot(y_te,kde=False, color = 'y')
    plt.show()
    
    rmse = np.sqrt(mean_squared_error(y_te, y_hat))
    xbg2_rmse_avg = xbg2_rmse_avg + rmse
    print('rmse: %5.3f'% rmse)
    
    y_hat_rd = np.round(y_hat)
    acc = accuracy_score(y_te,y_hat_rd)
    xbg2_acc_avg = xbg2_acc_avg +acc
    
    print('acc: %5.3f' % acc)
    
    y_hat_appro = y_hat_rd
    for i in range(len(y_hat)):  
        if(np.abs(y_hat_rd[i]-y_te[i])<=2):
            y_hat_appro[i]=y_te[i]
        else:
            y_hat_appro[i] = y_hat_rd[i]
            
    acc_appo = accuracy_score(y_te,y_hat_appro)
    xbg2_acc_appo = xbg2_acc_appo + acc_appo
    print('acc_appo: %5.3f'% acc_appo)
    
print('rmse_avg: %5.3f'% (xbg2_rmse_avg/5))
print('xbg2_acc_avg: %5.3f'% (xbg2_acc_avg/5))
print('xbg2_acc_appo: %5.3f'% (xbg2_acc_appo/5))
#%%
#xbg--Testing
y2_hat = xg2_reg.predict(X2_test_xbg)
sns.distplot(y2_hat,kde=False,color = 'skyblue')
sns.distplot(y2_test_xbg,kde=False, color = 'y')
plt.show()

y2_hat_rd = np.round(y2_hat)
acc2 = len(np.argwhere(y2_hat_rd == y2_test_xbg))/len(y2_test_xbg)
print('acc: %5.3f' % acc2)

count = 0
for i in range(len(y2_hat_rd)):
    if(np.absolute(y2_hat_rd[i]-y2_test_xbg[i])<=5):
        count = count+1
        
acc5_man = count/len(y2_hat_rd)
print('acc5_man: %5.3f' % acc5_man)

acc2_appo = len(np.argwhere(np.absolute(y2_hat_rd-y2_test_xbg<=2)))/len(y2_test_xbg)      
acc5_appo = len(np.argwhere(np.absolute(y2_hat_rd-y2_test_xbg<=5)))/len(y2_test_xbg)
        
#rmse2_appo_xbg = np.sqrt(mean_squared_error(y2_test_xbg, y2_hat_appro_xbg))
print('acc2: %5.3f'% acc2_appo)
print('acc5: %5.3f'% acc5_appo)
#%%
import matplotlib.pyplot as plt
xgb.plot_tree(xg2_reg)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()