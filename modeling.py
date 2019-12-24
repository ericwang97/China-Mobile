#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd #数据分析
import numpy as np #科学计算
from sklearn import linear_model
import xgboost as xgb
import joblib
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn import model_selection
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import statsmodels.api as sm
import processing
import model_measurement
import NN
from sklearn.externals import joblib


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


#########################################################################
#########################################################################
# 原始数据LR

def logistics_mod(X_train,Y_train,X_test,Y_test,drop_feature,combined,show_result,show_COR):

    if show_COR:
        model_measurement.plotCOR(X_train, WOE=False)
    else:
        pass

    logistics_mod = linear_model.LogisticRegression(solver='liblinear',C=1.0, tol=1e-6,verbose=0,fit_intercept=True)#,penalty='l2')

    if combined:
        X_train1=X_train.drop(drop_feature+['主叫_离散率组合_(1.0, 1.0)', '归属地是否未知_0.0'],axis=1)
        X_test1 = X_test.drop(drop_feature+['主叫_离散率组合_(1.0, 1.0)', '归属地是否未知_0.0'], axis=1)
    else:
        X_train1 = X_train.drop(drop_feature, axis=1)
        X_test1 = X_test.drop(drop_feature, axis=1)

    logistics_mod.fit(X_train1, Y_train)
    predictions = logistics_mod.predict_proba(X_test1)#每一类的概率
    model_measurement.plotROC(predictions, Y_test, 'LR', show=show_result)

    if show_result:
        model_measurement.VIF(X_train1, name='X_train1', combined=combined, WOE=False)
        model_measurement.plotKS(predictions,Y_test,'LR')
        model_measurement.coef(logistics_mod,X_train1,Y_train)

    else:
        pass

##################################################################
#SMOTE 数据+LR

def logistics_mod_SMOTE(X_train,Y_train,X_test,Y_test,drop_feature,minority,combined,show_result):
    X_train_SMOTE, Y_train_SMOTE, Transdata_train_SMOTE = processing.data_SMOTE(X_train,
                                                                                Y_train, minority= minority, combined= combined, WOE = False)

    print(Transdata_train_SMOTE.groupby(['是否是诈骗电话']).count())
    logistics_mod_SMOTE = linear_model.LogisticRegression(solver='liblinear',C=1.0, tol=1e-6,verbose=0)#,penalty='l2')
    if combined:
        X_train1_SMOTE = Transdata_train_SMOTE.drop(['是否是诈骗电话','主叫_离散率组合_(1.0, 1.0)', '归属地是否未知_0.0'] + drop_feature, axis=1)
        X_test1 = X_test.drop(drop_feature+['主叫_离散率组合_(1.0, 1.0)', '归属地是否未知_0.0'], axis=1)
    else:
        X_train1_SMOTE = Transdata_train_SMOTE.drop(['是否是诈骗电话'] + drop_feature, axis=1)
        X_test1 = X_test.drop(drop_feature, axis=1)
    logistics_mod_SMOTE.fit(X_train1_SMOTE, Y_train_SMOTE)
    predictions_SMOTE = logistics_mod_SMOTE.predict_proba(X_test1)#每一类的概率
    model_measurement.plotROC(predictions_SMOTE,Y_test,'LR+SMOTE',show=show_result)

    if show_result:
        model_measurement.plotKS(predictions_SMOTE, Y_test, 'LR+SMOTE')
        model_measurement.VIF(X_train1_SMOTE, name='X_train_SMOTE1', combined=combined, WOE=False)
        model_measurement.coef(logistics_mod_SMOTE, X_train1_SMOTE, Y_train_SMOTE)
    else:
        pass

######################################################################
# WOE数据+LR

def logistics_mod_woe(X_woe_train,X_woe_test,Y_woe_train,Y_woe_test,drop_feature,combined,show_COR,show_result):

    if show_COR:
        model_measurement.plotCOR(X_woe_train, WOE=True)
    else:
        pass

    logistics_mod_woe = linear_model.LogisticRegression(solver='liblinear',C=1.0, tol=1e-6,verbose=0,fit_intercept=True)#,penalty='l2')
    X_woe_train1=X_woe_train.drop(drop_feature,axis=1)                                                                   #去掉IV值低于0.02的几个变量，没有用
    #X_woe_train1 = X_woe_train.filter(regex='主叫次数|平均通话时间')

    logistics_mod_woe.fit(X_woe_train1, Y_woe_train)
    #joblib.dump(logistics_mod_woe, "logistics_mod_woe.m")
    #print('model is saved')

    X_woe_test1=X_woe_test.drop(drop_feature,axis=1)
    #logistics_mod_woe2 = joblib.load("logistics_mod_woe.m")
    #print('model is loaded')
    predictions_woe = logistics_mod_woe.predict_proba(X_woe_test1)#每一类的概率
    #predictions_woe1 = logistics_mod_woe.predict(X_woe_test1.values)
    model_measurement.plotROC(predictions_woe, Y_woe_test, 'LR+WOE', show=show_result)


    if show_result:
        model_measurement.VIF(X_woe_train1, name='X_woe_train1', combined=combined, WOE=True)
        model_measurement.plotKS(predictions_woe, Y_woe_test, 'LR+WOE')
        model_measurement.coef(logistics_mod_woe, X_woe_train1, Y_woe_train)
        # model_measurement.plot_learning_curve(logistics_mod_woe,'Learning Curve of Logistics',X_woe, Y_woe)

    else:
        pass

##################################################################
# WOE + SMOTE 数据 + LR

def logistics_mod_woe_SMOTE(X_woe_train,X_woe_test,Y_woe_train,Y_woe_test,drop_feature,minority,combined,show_result):
    X_woe_train_SMOTE, Y_woe_train_SMOTE, Transdata_woe_train_SMOTE = processing.data_SMOTE(X_woe_train, Y_woe_train, minority=minority, combined=combined, WOE = True)
    print(Transdata_woe_train_SMOTE.groupby(['是否是诈骗电话']).count())
    logistics_mod_woe_SMOTE = linear_model.LogisticRegression(solver='liblinear',C=1.0, tol=1e-6,verbose=0)#,penalty='l2')
    X_woe_train_SMOTE1=Transdata_woe_train_SMOTE.drop(['是否是诈骗电话']+ drop_feature,axis=1)
    logistics_mod_woe_SMOTE.fit(X_woe_train_SMOTE1, Y_woe_train_SMOTE)
    X_woe_test1=X_woe_test.drop(drop_feature,axis=1)
    predictions_woe_SMOTE = logistics_mod_woe_SMOTE.predict_proba(X_woe_test1)#每一类的概率
    model_measurement.plotROC(predictions_woe_SMOTE,Y_woe_test,'LR+WOE+SMOTE',show=show_result)
    if show_result:
        model_measurement.plotKS(predictions_woe_SMOTE, Y_woe_test, 'LR+WOE+SMOTE')
        model_measurement.VIF(X_woe_train_SMOTE1, name='X_woe_train_SMOTE1', combined=combined, WOE=True)
    else:
        pass

##################################################################
# WOE + SMOTE 数据 + SVM

def SVM(X_train,Y_train,X_test,Y_test,kernel,C,drop_feature,show_result):

    grid_svc_best = svm.SVC(C = C, kernel=kernel,gamma='auto')
    X_train1 = X_train.drop(drop_feature, axis=1)
    grid_svc_best.fit(X_train1.values, Y_train.values)
    X_test1 = X_test.drop(drop_feature, axis=1)
    pred_SVM = grid_svc_best.predict(X_test1.values)
    model_measurement.plotROC(pred_SVM, Y_test, name='SVM',method='SVM',show=show_result)

def SVM_grid(X_train,Y_train,X_test,Y_test,grid_kernel,grid_C,n_jobs,drop_feature,show_result):

    # 使用网格搜索法，选择非线性可分SVM“类”中的最佳C值和核函数
    parameters = {'kernel': grid_kernel, 'C': grid_C}
    grid_svc = model_selection.GridSearchCV(estimator=svm.SVC(gamma = 'auto'),
                                            param_grid=parameters,
                                            scoring='roc_auc', cv=5, verbose=1,n_jobs=n_jobs)
    # 模型在训练数据集上的拟合
    X_train1 = X_train.drop(drop_feature, axis=1)
    grid_svc.fit(X_train1, Y_train)
    print('the best params:{}, the best training AUC:{:0.3f}'.format(grid_svc.best_params_, grid_svc.best_score_))
    grid_svc_best = svm.SVC(C = grid_svc.best_params_['C'], kernel=grid_svc.best_params_['kernel'],gamma='auto')
    X_train1 = X_train.drop(drop_feature, axis=1)
    grid_svc_best.fit(X_train1, Y_train)
    X_test1 = X_test.drop(drop_feature, axis=1)
    pred_SVM = grid_svc_best.predict(X_test1)
    model_measurement.plotROC(pred_SVM, Y_test, name='SVM(Grid)',method='SVM',show=show_result)

def SVM_SMOTE(X_train,Y_train,X_test,Y_test,kernel,C,drop_feature,minority,combined,show_result):

    X_train_SMOTE, Y_train_SMOTE, Transdata_train_SMOTE = processing.data_SMOTE(X_train, Y_train, minority=minority, combined=combined, WOE=False)

    grid_svc_best = svm.SVC(C = C, kernel=kernel,gamma='auto')

    X_train_SMOTE1 = Transdata_train_SMOTE.drop(['是否是诈骗电话'] + drop_feature, axis=1)
    grid_svc_best.fit(X_train_SMOTE1, Y_train_SMOTE)
    X_test1 = X_test.drop(drop_feature, axis=1)
    pred_SVM_SMOTE = grid_svc_best.predict(X_test1)
    model_measurement.plotROC(pred_SVM_SMOTE, Y_test, name='SVM+SMOTE',method='SVM+SMOTE',show=show_result)

def SVM_SMOTE_grid(X_train,Y_train,X_test,Y_test,grid_kernel,grid_C,n_jobs,drop_feature,minority,combined,show_result):

    X_train_SMOTE, Y_train_SMOTE, Transdata_train_SMOTE = processing.data_SMOTE(X_train, Y_train, minority=minority, combined=combined, WOE=False)

    # 使用网格搜索法，选择非线性可分SVM“类”中的最佳C值和核函数
    parameters = {'kernel': grid_kernel, 'C': grid_C}
    grid_svc = model_selection.GridSearchCV(estimator=svm.SVC(gamma = 'auto'),
                                            param_grid=parameters,
                                            scoring='roc_auc', cv=5, verbose=1,n_jobs=n_jobs)
    # 模型在训练数据集上的拟合
    X_train_SMOTE1 = Transdata_train_SMOTE.drop(['是否是诈骗电话'] + drop_feature, axis=1)
    grid_svc.fit(X_train_SMOTE1, Y_train_SMOTE)
    print('the best params:{}, the best training AUC:{:0.3f}'.format(grid_svc.best_params_, grid_svc.best_score_))


    grid_svc_best = svm.SVC(C = grid_svc.best_params_['C'], kernel=grid_svc.best_params_['kernel'],gamma='auto')

    X_train_SMOTE1 = Transdata_train_SMOTE.drop(['是否是诈骗电话'] + drop_feature, axis=1)
    grid_svc_best.fit(X_train_SMOTE1, Y_train_SMOTE)
    X_test1 = X_test.drop(drop_feature, axis=1)
    pred_SVM_SMOTE = grid_svc_best.predict(X_test1)
    model_measurement.plotROC(pred_SVM_SMOTE, Y_test, name='SVM+SMOTE(Grid)',method='SVM+SMOTE',show=show_result)

def SVM_woe(X_woe_train,X_woe_test,Y_woe_train,Y_woe_test,kernel,C,drop_feature,show_result):

    grid_svc_best = svm.SVC(C = C, kernel=kernel,gamma='auto')

    X_woe_train1 = X_woe_train.drop(drop_feature, axis=1)
    grid_svc_best.fit(X_woe_train1, Y_woe_train)
    X_woe_test1 = X_woe_test.drop(drop_feature, axis=1)
    pred_SVM_woe = grid_svc_best.predict(X_woe_test1)
    model_measurement.plotROC(pred_SVM_woe, Y_woe_test, name='SVM+WOE',method='SVM+WOE',show=show_result)

def SVM_woe_grid(X_woe_train,X_woe_test,Y_woe_train,Y_woe_test,grid_kernel,grid_C,n_jobs,drop_feature,show_result):

    # 使用网格搜索法，选择非线性可分SVM“类”中的最佳C值和核函数
    parameters = {'kernel': grid_kernel, 'C': grid_C}
    grid_svc = model_selection.GridSearchCV(estimator=svm.SVC(gamma = 'auto'),
                                            param_grid=parameters,
                                            scoring='roc_auc', cv=5, verbose=1,n_jobs=n_jobs)
    # 模型在训练数据集上的拟合
    X_woe_train1 = X_woe_train.drop(drop_feature, axis=1)
    grid_svc.fit(X_woe_train1, Y_woe_train)
    print('the best params:{}, the best training AUC:{:0.3f}'.format(grid_svc.best_params_, grid_svc.best_score_))
    grid_svc_best = svm.SVC(C = grid_svc.best_params_['C'], kernel=grid_svc.best_params_['kernel'],gamma='auto')

    X_woe_train1 = X_woe_train.drop(drop_feature, axis=1)
    grid_svc_best.fit(X_woe_train1, Y_woe_train)
    X_woe_test1 = X_woe_test.drop(drop_feature, axis=1)
    pred_SVM_woe = grid_svc_best.predict(X_woe_test1)
    model_measurement.plotROC(pred_SVM_woe, Y_woe_test, name='SVM+WOE(Grid)',method='SVM+WOE',show=show_result)

def SVM_woe_SMOTE(X_woe_train,X_woe_test,Y_woe_train,Y_woe_test,kernel,C,drop_feature,minority,combined,show_result):

    X_woe_train_SMOTE, Y_woe_train_SMOTE, Transdata_woe_train_SMOTE = processing.data_SMOTE(X_woe_train, Y_woe_train, minority=minority, combined=combined, WOE=True)
    grid_svc_best = svm.SVC(C = C, kernel=kernel,gamma='auto')

    X_woe_train_SMOTE1 = Transdata_woe_train_SMOTE.drop(['是否是诈骗电话'] + drop_feature, axis=1)
    grid_svc_best.fit(X_woe_train_SMOTE1, Y_woe_train_SMOTE)
    X_woe_test1 = X_woe_test.drop(drop_feature, axis=1)
    pred_SVM_woe_SMOTE = grid_svc_best.predict(X_woe_test1)
    model_measurement.plotROC(pred_SVM_woe_SMOTE, Y_woe_test, name='SVM+WOE+SMOTE',method='SVM+WOE+SMOTE',show=show_result)

def SVM_woe_SMOTE_grid(X_woe_train,X_woe_test,Y_woe_train,Y_woe_test,grid_kernel,grid_C,n_jobs,drop_feature,minority,combined,show_result):

    X_woe_train_SMOTE, Y_woe_train_SMOTE, Transdata_woe_train_SMOTE = processing.data_SMOTE(X_woe_train, Y_woe_train, minority=minority, combined=combined, WOE=True)

    # 使用网格搜索法，选择非线性可分SVM“类”中的最佳C值和核函数
    parameters = {'kernel': grid_kernel, 'C': grid_C}
    grid_svc = model_selection.GridSearchCV(estimator=svm.SVC(gamma = 'auto'),
                                            param_grid=parameters,
                                            scoring='roc_auc', cv=5, verbose=1,n_jobs=n_jobs)
    # 模型在训练数据集上的拟合
    X_woe_train_SMOTE1 = Transdata_woe_train_SMOTE.drop(['是否是诈骗电话'] + drop_feature, axis=1)
    grid_svc.fit(X_woe_train_SMOTE1, Y_woe_train_SMOTE)
    print('the best params:{}, the best training AUC:{:0.3f}'.format(grid_svc.best_params_, grid_svc.best_score_))


    grid_svc_best = svm.SVC(C = grid_svc.best_params_['C'], kernel=grid_svc.best_params_['kernel'],gamma='auto')

    X_woe_train_SMOTE1 = Transdata_woe_train_SMOTE.drop(['是否是诈骗电话'] + drop_feature, axis=1)
    grid_svc_best.fit(X_woe_train_SMOTE1, Y_woe_train_SMOTE)
    X_woe_test1 = X_woe_test.drop(drop_feature, axis=1)
    pred_SVM_woe_SMOTE = grid_svc_best.predict(X_woe_test1)
    model_measurement.plotROC(pred_SVM_woe_SMOTE, Y_woe_test, name='SVM+WOE+SMOTE(Grid)',method='SVM+WOE+SMOTE',show=show_result)

##########################################################################
# XGBoost数据
def mod_xgb(X_train,Y_train,X_test,Y_test,show_result,nthread, learning_rate, n_estimators, max_depth, gamma,subsample, colsample_bytree):
    mod_xgb = xgb.XGBClassifier(nthread=nthread,     #含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                              learning_rate=learning_rate,    #含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                              n_estimators=n_estimators,       #含义：总共迭代的次数，即决策树的个数
                              max_depth=max_depth,           #含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                              gamma=gamma,               #含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                              subsample=subsample,       #含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                              colsample_bytree=colsample_bytree) #训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。

    mod_xgb.fit(X_train.values, Y_train.values)

    predictions_xgb = mod_xgb.predict_proba(X_test.values)#每一类的概率
    predictions_xgb1 = mod_xgb.predict(X_test.values)
    model_measurement.plotROC(predictions_xgb,Y_test,name='XGB',show=show_result)
    #model_measurement.plotKS(predictions_xgb,Y_test,'XGB')

def mod_xgb_SMOTE(X_train,Y_train,X_test,Y_test,minority,combined,show_result,nthread, learning_rate, n_estimators, max_depth, gamma,subsample, colsample_bytree):
    X_train_SMOTE, Y_train_SMOTE, Transdata_train_SMOTE = processing.data_SMOTE(X_train, Y_train, minority=minority,
                                                                                combined=combined, WOE=False)
    mod_xgb_SMOTE = xgb.XGBClassifier(nthread=nthread,  #含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                                      learning_rate=learning_rate, # 含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                                      n_estimators=n_estimators,  # 含义：总共迭代的次数，即决策树的个数
                                      max_depth=max_depth,  # 含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                                      gamma=gamma,  # 含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                                      subsample=subsample,  # 含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                                      colsample_bytree=colsample_bytree)  # 训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。

    mod_xgb_SMOTE.fit(X_train_SMOTE, Y_train_SMOTE)

    predictions_xgb_SMOTE = mod_xgb_SMOTE.predict_proba(X_test.values)#每一类的概率
    model_measurement.plotROC(predictions_xgb_SMOTE,Y_test,name='XGB+SMOTE',show=show_result)

def mod_xgb_woe(X_woe_train,Y_woe_train,X_woe_test,Y_woe_test,show_result,nthread, learning_rate, n_estimators, max_depth, gamma,subsample, colsample_bytree):
    mod_xgb_woe = xgb.XGBClassifier(nthread=nthread,  #含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                                    learning_rate=learning_rate, # 含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                                    n_estimators=n_estimators,  # 含义：总共迭代的次数，即决策树的个数
                                    max_depth=max_depth,  # 含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                                    gamma=gamma,  # 含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                                    subsample=subsample,  # 含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                                    colsample_bytree=colsample_bytree)  # 训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。

    mod_xgb_woe.fit(X_woe_train.values, Y_woe_train.values)

    predictions_xgb_WOE = mod_xgb_woe.predict_proba(X_woe_test.values)#每一类的概率
    predictions_xgb_WOE1 = mod_xgb_woe.predict(X_woe_test.values)
    model_measurement.plotROC(predictions_xgb_WOE,Y_woe_test,name='XGB+WOE',show=show_result)
    #model_measurement.plotKS(predictions_xgb,Y_test,'XGB')

def mod_xgb_woe_SMOTE(X_woe_train,Y_woe_train,X_woe_test,Y_woe_test,minority,combined,show_result,nthread, learning_rate, n_estimators, max_depth, gamma,subsample, colsample_bytree):
    X_woe_train_SMOTE, Y_woe_train_SMOTE, Transdata_woe_train_SMOTE = processing.data_SMOTE(X_woe_train, Y_woe_train,
                                                                                            minority=minority,
                                                                                            combined=combined, WOE=True)
    mod_xgb_woe_SMOTE = xgb.XGBClassifier(nthread=nthread,  #含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                                          learning_rate=learning_rate, # 含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                                          n_estimators=n_estimators,  # 含义：总共迭代的次数，即决策树的个数
                                          max_depth=max_depth,  # 含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                                          gamma=gamma,  # 含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                                          subsample=subsample,
                                          # 含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                                          colsample_bytree=colsample_bytree)  # 训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
    mod_xgb_woe_SMOTE.fit(X_woe_train_SMOTE, Y_woe_train_SMOTE)

    predictions_xgb_WOE_SMOTE = mod_xgb_woe_SMOTE.predict_proba(X_woe_test.values)#每一类的概率
    model_measurement.plotROC(predictions_xgb_WOE_SMOTE,Y_woe_test,name='XGB+WOE+SMOTE',show=show_result)

##################################################
# XGB + LR
# xgboost 编码原有特征, apply()方法可以获得leaf indices(叶节点索引)

def logistics_mod_xgb(X_train,Y_train,X_test,Y_test,show_result,nthread, learning_rate, n_estimators, max_depth, gamma,subsample, colsample_bytree):

    mod_xgb = xgb.XGBClassifier(nthread=nthread,     #含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                              learning_rate=learning_rate,    #含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                              n_estimators=n_estimators,       #含义：总共迭代的次数，即决策树的个数
                              max_depth=max_depth,           #含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                              gamma=gamma,               #含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                              subsample=subsample,       #含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                              colsample_bytree=colsample_bytree) #训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。

    mod_xgb.fit(X_train.values, Y_train.values)

    X_train_leaves = mod_xgb.apply(X_train.values)
    X_test_leaves = mod_xgb.apply(X_test.values)
    train_rows = X_train_leaves.shape[0]
    # 合并编码后的训练数据和测试数据
    X_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    X_leaves = X_leaves.astype(np.int32)

    XGBencoder = OneHotEncoder()
    X_trans = XGBencoder.fit_transform(X_leaves)
    X_train_xgb = hstack([X_trans[:train_rows, :], X_train.astype(float)])
    X_test_xgb = hstack([X_trans[train_rows:, :], X_test.astype(np.float64)])

    logistics_mod_xgb = linear_model.LogisticRegression(solver='liblinear',C=1.0, tol=1e-6)#,verbose=1)#penalty='l2'
    logistics_mod_xgb.fit(X_train_xgb, Y_train)
    predictions_xgblr = logistics_mod_xgb.predict_proba(X_test_xgb)
    #predictions_xgblr1 = logistics_mod_xgb.predict(X_test_xgb.values)
    model_measurement.plotROC(predictions_xgblr,Y_test,name='XGBlr',show=show_result)

def logistics_mod_xgb_SMOTE(X_train,Y_train,X_test,Y_test,minority,combined,show_result,nthread, learning_rate, n_estimators, max_depth, gamma,subsample, colsample_bytree):
    X_train_SMOTE, Y_train_SMOTE, Transdata_train_SMOTE = processing.data_SMOTE(X_train, Y_train, minority=minority,
                                                                                combined=combined, WOE=False)
    mod_xgb_SMOTE = xgb.XGBClassifier(nthread=nthread,  #含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                                      learning_rate=learning_rate, # 含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                                      n_estimators=n_estimators,  # 含义：总共迭代的次数，即决策树的个数
                                      max_depth=max_depth,  # 含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                                      gamma=gamma,  # 含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                                      subsample=subsample,  # 含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                                      colsample_bytree=colsample_bytree)  # 训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。

    mod_xgb_SMOTE.fit(X_train_SMOTE, Y_train_SMOTE)

    X_train_leaves = mod_xgb_SMOTE.apply(X_train_SMOTE)
    X_test_leaves = mod_xgb_SMOTE.apply(X_test.values)
    train_rows = X_train_leaves.shape[0]
    # 合并编码后的训练数据和测试数据
    X_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    X_leaves = X_leaves.astype(np.int32)

    XGBencoder = OneHotEncoder()
    X_trans = XGBencoder.fit_transform(X_leaves)
    X_train_xgb = hstack([X_trans[:train_rows, :], X_train_SMOTE.astype(float)])
    X_test_xgb = hstack([X_trans[train_rows:, :], X_test.astype(np.float64)])

    logistics_mod_xgb = linear_model.LogisticRegression(solver='liblinear',C=1.0, tol=1e-6)#,verbose=1)#penalty='l2'
    logistics_mod_xgb.fit(X_train_xgb, Y_train_SMOTE)
    predictions_xgblr_SMOTE = logistics_mod_xgb.predict_proba(X_test_xgb)
    #predictions_xgblr1 = logistics_mod_xgb.predict(X_test_xgb.values)
    model_measurement.plotROC(predictions_xgblr_SMOTE,Y_test,name='XGBlr+SMOTE',show=show_result)

def logistics_mod_xgb_woe(X_woe_train,Y_woe_train,X_woe_test,Y_woe_test,show_result,nthread, learning_rate, n_estimators, max_depth, gamma,subsample, colsample_bytree):

    mod_xgb_woe = xgb.XGBClassifier(nthread=nthread,  #含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                                    learning_rate=learning_rate, # 含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                                    n_estimators=n_estimators,  # 含义：总共迭代的次数，即决策树的个数
                                    max_depth=max_depth,  # 含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                                    gamma=gamma,  # 含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                                    subsample=subsample,  # 含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                                    colsample_bytree=colsample_bytree)  # 训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。

    mod_xgb_woe.fit(X_woe_train.values, Y_woe_train.values)

    X_train_leaves = mod_xgb_woe.apply(X_woe_train.values)
    X_test_leaves = mod_xgb_woe.apply(X_woe_test.values)
    train_rows = X_train_leaves.shape[0]
    # 合并编码后的训练数据和测试数据
    X_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    X_leaves = X_leaves.astype(np.int32)

    XGBencoder = OneHotEncoder()
    X_trans = XGBencoder.fit_transform(X_leaves)
    X_train_xgb = hstack([X_trans[:train_rows, :], X_woe_train.astype(float)])
    X_test_xgb = hstack([X_trans[train_rows:, :], X_woe_test.astype(np.float64)])

    logistics_mod_xgb = linear_model.LogisticRegression(solver='liblinear',C=1.0, tol=1e-6)#,verbose=1)#penalty='l2'
    logistics_mod_xgb.fit(X_train_xgb, Y_woe_train)
    predictions_xgblr = logistics_mod_xgb.predict_proba(X_test_xgb)
    #predictions_xgblr1 = logistics_mod_xgb.predict(X_test_xgb.values)
    model_measurement.plotROC(predictions_xgblr,Y_woe_test,name='XGBlr+WOE',show=show_result)
    #model_measurement.plotKS(predictions_xgblr,Y_test,'LR+XGB')

def logistics_mod_xgb_woe_SMOTE(X_woe_train,Y_woe_train,X_woe_test,Y_woe_test,minority,combined,show_result,nthread, learning_rate, n_estimators, max_depth, gamma,subsample, colsample_bytree):
    X_woe_train_SMOTE, Y_woe_train_SMOTE, Transdata_woe_train_SMOTE = processing.data_SMOTE(X_woe_train, Y_woe_train,
                                                                                            minority=minority,
                                                                                            combined=combined, WOE=True)
    mod_xgb_woe_SMOTE = xgb.XGBClassifier(nthread=nthread,  #含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                                          learning_rate=learning_rate, # 含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                                          n_estimators=n_estimators,  # 含义：总共迭代的次数，即决策树的个数
                                          max_depth=max_depth,  # 含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                                          gamma=gamma,  # 含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                                          subsample=subsample,
                                          # 含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                                          colsample_bytree=colsample_bytree)  # 训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
    mod_xgb_woe_SMOTE.fit(X_woe_train_SMOTE, Y_woe_train_SMOTE)

    X_train_leaves = mod_xgb_woe_SMOTE.apply(X_woe_train_SMOTE)
    X_test_leaves = mod_xgb_woe_SMOTE.apply(X_woe_test.values)
    train_rows = X_train_leaves.shape[0]
    # 合并编码后的训练数据和测试数据
    X_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    X_leaves = X_leaves.astype(np.int32)

    XGBencoder = OneHotEncoder()
    X_trans = XGBencoder.fit_transform(X_leaves)
    X_train_xgb = hstack([X_trans[:train_rows, :], X_woe_train_SMOTE.astype(float)])
    X_test_xgb = hstack([X_trans[train_rows:, :], X_woe_test.astype(np.float64)])

    logistics_mod_xgb = linear_model.LogisticRegression(solver='liblinear', C=1.0, tol=1e-6)  # ,verbose=1)#penalty='l2'
    logistics_mod_xgb.fit(X_train_xgb, Y_woe_train_SMOTE)
    predictions_xgblr = logistics_mod_xgb.predict_proba(X_test_xgb)
    model_measurement.plotROC(predictions_xgblr, Y_woe_test, name='XGBlr+WOE+SMOTE', show=show_result)


##########################################################################
# NN方法

def NN_mod(X_train,Y_train,X_test,Y_test,loss,intermediate_dim,epochs,batch_size,dropout,activation,output_activation,show_result):

    X_train_NN, X_test_NN, Y_train_NN, Y_test_NN = NN.NN_processing(X_train.values, X_test.values, Y_train.values, Y_test.values)
    model = NN.NN_main_func(original_dim=X_train_NN.shape[1],intermediate_dim=intermediate_dim,loss=loss,dropout = dropout,activation=activation,output_activation=output_activation)
    model.fit(X_train_NN, Y_train_NN, epochs = epochs, batch_size = batch_size, validation_data = (X_test_NN, Y_test_NN),verbose=show_result)
    predictions_NN = model.predict(X_test_NN)
    model_measurement.plotROC(predictions_NN,Y_test_NN,show=show_result,name='NN',method='NN')
    #model_measurement.plotKS(predictions_NN,Y_test_NN,'NN','NN')

def NN_SMOTE_mod(X_train,Y_train,X_test,Y_test,minority,combined,loss,intermediate_dim,epochs,batch_size,dropout,activation,output_activation,show_result):

    X_train_SMOTE, Y_train_SMOTE, Transdata_train_SMOTE = processing.data_SMOTE(X_train, Y_train, minority=minority,
                                                                                combined=combined, WOE=False)
    X_train_NN, X_test_NN, Y_train_NN, Y_test_NN = NN.NN_processing(X_train_SMOTE, X_test.values, Y_train_SMOTE, Y_test.values)
    model = NN.NN_main_func(X_train_NN.shape[1],intermediate_dim,loss=loss,dropout = dropout,activation=activation,output_activation=output_activation)
    model.fit(X_train_NN, Y_train_NN, epochs = epochs, batch_size = batch_size, validation_data = (X_test_NN, Y_test_NN),verbose=show_result)
    predictions_NN = model.predict(X_test_NN)
    model_measurement.plotROC(predictions_NN,Y_test_NN,show=show_result,name='NN+SMOTE',method='NN')

def NN_woe_mod(X_woe_train,Y_woe_train, X_woe_test,Y_woe_test,loss,intermediate_dim,epochs,batch_size,dropout,activation,output_activation,show_result):

    X_woe_train_NN, X_woe_test_NN, Y_woe_train_NN, Y_woe_test_NN = NN.NN_processing(X_woe_train.values, X_woe_test.values, Y_woe_train.values, Y_woe_test.values)
    model = NN.NN_main_func(X_woe_train_NN.shape[1],intermediate_dim,loss=loss,dropout = dropout,activation= activation,output_activation=output_activation)
    model.fit(X_woe_train_NN, Y_woe_train_NN, epochs = epochs, batch_size = batch_size, validation_data = (X_woe_test_NN, Y_woe_test_NN),verbose=show_result)
    predictions_NN = model.predict(X_woe_test_NN)
    model_measurement.plotROC(predictions_NN,Y_woe_test_NN,show=show_result,name='NN+WOE',method='NN')

def NN_woe_SMOTE_mod(X_woe_train, Y_woe_train, X_woe_test, Y_woe_test, minority,combined,loss, intermediate_dim, epochs, batch_size, dropout,activation, output_activation,show_result):

    X_train_SMOTE, Y_train_SMOTE, Transdata_train_SMOTE = processing.data_SMOTE(X_woe_train, Y_woe_train, minority=minority, combined=combined, WOE=True)
    X_woe_train_NN, X_woe_test_NN, Y_woe_train_NN, Y_woe_test_NN = NN.NN_processing(X_train_SMOTE, X_woe_test.values, Y_train_SMOTE, Y_woe_test.values)
    model = NN.NN_main_func(X_woe_train_NN.shape[1], intermediate_dim, loss=loss, dropout=dropout,
                            activation=activation,output_activation=output_activation)
    model.fit(X_woe_train_NN, Y_woe_train_NN, epochs=epochs, batch_size=batch_size,
              validation_data=(X_woe_test_NN, Y_woe_test_NN), verbose=show_result)
    predictions_NN = model.predict(X_woe_test_NN)
    model_measurement.plotROC(predictions_NN, Y_woe_test_NN, show=show_result, name='NN+WOE+SMOTE', method='NN')