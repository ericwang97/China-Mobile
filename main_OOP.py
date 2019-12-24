# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 2019

@author: Liwei Wang

"""

import pandas as pd #数据分析
import numpy as np #科学计算
import processing
import modeling
from WOE_Transformation import WOE_main_func
import warnings
warnings.filterwarnings("ignore")

###########################################

#[path]
path = {'X_path':"D:/力为的文件！！！/我的大学/移动实习/数据及建模/带标签数据/train.txt",
        'Y_path':"D:/力为的文件！！！/我的大学/移动实习/数据及建模/带标签数据/train_result_sha.txt"}

#[data_processing]
Newphone = ['小米','华为','苹果','OPPO','VIVO','三星','魅族']
worktime = np.append(range(8,12+1),range(14,20+1))                                                                      #定在8~20点工作时间是最好的
least_time = 7

#[train_test_split]
random_state = 50
test_size = 0.3
minority = 5000

#[WOE]
#  ['主叫次数', '平均通话时间', '被叫所属地个数', '通话失败率', '平均呼叫离散率', '呼叫银行次数', '使用旧款手机次数', '归属地是否未知','工作非工作时间通话差值','主叫_离散率组合']
methods = ['Tree','Tree','Kmeans','Tree','Tree','Kmeans','Tree','Tree','Tree']
group_num = [3,4,3,4,3,2,4,2,3]

#[SVM]
grid_kernel = ['rbf', 'poly', 'sigmoid']
grid_C=[0.1, 0.5, 1, 1.5, 2]
kernel = 'rbf'
C = 2

#[NN]
intermediate_dim=[128,64]

#[combined('主叫次数', '平均呼叫离散率'),re_index,name of re_index,dummy_feature,drop feature]
combined = True
if combined:
    re_index = True
    row_name = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    dummy_feature = ['主叫_离散率组合', '归属地是否未知']
    drop_feature = ['使用旧款手机次数', '呼叫银行次数', '主叫次数', '平均呼叫离散率']
else:
    re_index = False
    row_name = None
    dummy_feature = None
    drop_feature = ['使用旧款手机次数', '呼叫银行次数']

#[show]
show_WOE = False
show_result = False
show_COR = False

#######################################################################

class Telephone_Fraud_Data_Processor(pd.DataFrame):

    def data_processing(self, X, Y, worktime, NewPhone, least_time, combined=False):

        return processing.data_processing(X, Y, worktime, NewPhone, least_time, combined=combined)

    def WOE_transformation(self,data,methods,group_num,show_WOE = False,combined = False,re_index = False,row_name = None):

        return WOE_main_func(data=data,methods=methods, group_num=group_num, show_WOE=show_WOE, combined=combined, re_index=re_index, row_name=row_name)

class Telephone_Fraud_Data_Modeling():

    def __init__(self,data,WOE,drop_feature,dummy_feature,combined,re_index,row_name,random_state,test_size,show_result,show_COR,show_WOE):
        self.data = data
        self.WOE = WOE
        self.drop_feature = drop_feature
        self.combined = combined
        self.re_index = re_index
        self.row_name = row_name
        self.random_state = random_state
        self.test_size = test_size
        self.dummy_feature = dummy_feature
        self.show_result = show_result
        self.show_WOE = show_WOE
        self.show_COR = show_COR

    def train_split(self):

        return processing.train_split(data = self.data,random_state=self.random_state, test_size=self.test_size, WOE=self.WOE, combined=self.combined, dummy_feature=self.dummy_feature)

    def LR(self,SMOTE=False,SMOTE_minority=0):

        WOE = self.WOE

        if WOE:
            X_woe, Y_woe, X_woe_train, X_woe_test, Y_woe_train, Y_woe_test, Transdata_woe_train, Transdata_woe_test = self.train_split()

            if SMOTE:

                modeling.logistics_mod_woe_SMOTE(X_woe_train, X_woe_test, Y_woe_train, Y_woe_test,
                                                 drop_feature=self.drop_feature, minority=SMOTE_minority,
                                                 combined=self.combined, show_result=self.show_result)
            else:
                modeling.logistics_mod_woe(X_woe_train, X_woe_test, Y_woe_train, Y_woe_test, drop_feature=self.drop_feature,
                                           combined=self.combined, show_result=self.show_result,show_COR=self.show_COR)

        else:

            X, Y, X_train, X_test, Y_train, Y_test, Transdata_train, Transdata_test = self.train_split()

            if SMOTE:

                modeling.logistics_mod_SMOTE(X_train, Y_train, X_test, Y_test, minority=SMOTE_minority,
                                             drop_feature=self.drop_feature, combined=self.combined,
                                             show_result=self.show_result)
            else:

                modeling.logistics_mod(X_train, Y_train, X_test, Y_test, drop_feature=self.drop_feature, combined=self.combined, show_COR=self.show_COR,show_result=self.show_result)

    def SVM(self,grid=False,C=None,kernel = None,grid_kernel=None, grid_C=None, n_jobs=4,SMOTE=False,SMOTE_minority=0):

        WOE = self.WOE

        if WOE:
            X_woe, Y_woe, X_woe_train, X_woe_test, Y_woe_train, Y_woe_test, Transdata_woe_train, Transdata_woe_test = self.train_split()

            if SMOTE:

                if grid:

                    modeling.SVM_woe_SMOTE_grid(X_woe_train=X_woe_train, Y_woe_train=Y_woe_train,X_woe_test=X_woe_test, Y_woe_test=Y_woe_test,grid_kernel=grid_kernel, grid_C=grid_C, n_jobs=n_jobs,
                                           drop_feature=self.drop_feature, minority=SMOTE_minority, combined=self.combined, show_result=self.show_result)
                else:
                    modeling.SVM_woe_SMOTE(X_woe_train=X_woe_train, Y_woe_train=Y_woe_train,X_woe_test=X_woe_test, Y_woe_test=Y_woe_test, kernel=kernel, C=C, drop_feature=drop_feature, minority=SMOTE_minority, combined=self.combined,show_result=show_result)

            else:

                if grid:

                    modeling.SVM_woe_grid(X_woe_train=X_woe_train, Y_woe_train=Y_woe_train,X_woe_test=X_woe_test, Y_woe_test=Y_woe_test,
                                       grid_kernel=grid_kernel, grid_C=grid_C, n_jobs=n_jobs,
                                       drop_feature=self.drop_feature,show_result=self.show_result)
                else:
                    modeling.SVM_woe(X_woe_train=X_woe_train, Y_woe_train=Y_woe_train,X_woe_test=X_woe_test, Y_woe_test=Y_woe_test, kernel=kernel, C=C,
                                 drop_feature=drop_feature, show_result=show_result)

        else:

            X, Y, X_train, X_test, Y_train, Y_test, Transdata_train, Transdata_test = self.train_split()

            if SMOTE:

                if grid:

                    modeling.SVM_SMOTE_grid(
                                       grid_kernel=grid_kernel, grid_C=grid_C, n_jobs=n_jobs,X_train=X_train,X_test= X_test, Y_train=Y_train, Y_test=Y_test,
                                       drop_feature=self.drop_feature, minority=SMOTE_minority, combined=self.combined,
                                       show_result=self.show_result)
                else:

                    modeling.SVM_SMOTE(X_train=X_train,X_test= X_test, Y_train=Y_train, Y_test=Y_test,  C=C, kernel=kernel,
                                       drop_feature=self.drop_feature, minority=SMOTE_minority, combined=self.combined,
                                       show_result=self.show_result)
            else:

                if grid:

                    modeling.SVM_grid(X_train=X_train,X_test= X_test, Y_train=Y_train, Y_test=Y_test,
                                 grid_kernel=grid_kernel, grid_C=grid_C, n_jobs=n_jobs,
                                 drop_feature=self.drop_feature, show_result=self.show_result)
                else:
                    modeling.SVM(X_train=X_train,X_test= X_test, Y_train=Y_train, Y_test=Y_test, C=C, kernel=kernel,
                                 drop_feature=self.drop_feature, show_result=self.show_result)

    def XGB(self,SMOTE=False,SMOTE_minority=0,nthread=4,learning_rate=0.1,n_estimators=50,max_depth=6,gamma=0, subsample=0.9,colsample_bytree=0.7):

        WOE = self.WOE

        if WOE:

            X_woe, Y_woe, X_woe_train, X_woe_test, Y_woe_train, Y_woe_test, Transdata_woe_train, Transdata_woe_test = self.train_split()

            if SMOTE:

                modeling.mod_xgb_woe_SMOTE(X_woe_train, Y_woe_train, X_woe_test, Y_woe_test, show_result=self.show_result,minority=SMOTE_minority,combined=self.combined,
                                           nthread=nthread, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                                           gamma=gamma,subsample=subsample, colsample_bytree=colsample_bytree)

            else:

                modeling.mod_xgb_woe(X_woe_train, Y_woe_train, X_woe_test, Y_woe_test, show_result=self.show_result,
                                     nthread=nthread, learning_rate=learning_rate, n_estimators=n_estimators,max_depth=max_depth,
                                     gamma=gamma, subsample=subsample,colsample_bytree=colsample_bytree)

        else:

            X, Y, X_train, X_test, Y_train, Y_test, Transdata_train, Transdata_test = self.train_split()

            if SMOTE:

                modeling.mod_xgb_SMOTE(X_train, Y_train, X_test, Y_test, show_result=self.show_result,minority=SMOTE_minority,combined=self.combined,
                                       nthread=nthread, learning_rate=learning_rate, n_estimators=n_estimators,
                                       max_depth=max_depth, gamma=gamma, subsample=subsample,
                                       colsample_bytree=colsample_bytree)

            else:

                modeling.mod_xgb(X_train, Y_train, X_test, Y_test, show_result=self.show_result,
                                 nthread=nthread, learning_rate=learning_rate, n_estimators=n_estimators,
                                 max_depth=max_depth, gamma=gamma, subsample=subsample,
                                 colsample_bytree=colsample_bytree)

    def XGBLR(self,SMOTE=False,SMOTE_minority=0,nthread=4,learning_rate=0.1,n_estimators=50,max_depth=6,gamma=0, subsample=0.9,colsample_bytree=0.7):

        WOE = self.WOE

        if WOE:
            X_woe, Y_woe, X_woe_train, X_woe_test, Y_woe_train, Y_woe_test, Transdata_woe_train, Transdata_woe_test = self.train_split()

            if SMOTE:

                modeling.logistics_mod_xgb_woe_SMOTE(X_woe_train,Y_woe_train, X_woe_test,Y_woe_test, show_result=self.show_result,minority=SMOTE_minority,combined=self.combined,
                                           nthread=nthread, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                                           gamma=gamma,subsample=subsample, colsample_bytree=colsample_bytree)

            else:

                modeling.logistics_mod_xgb_woe(X_woe_train, Y_woe_train, X_woe_test, Y_woe_test,show_result=self.show_result,nthread=nthread, learning_rate=learning_rate, n_estimators=n_estimators,max_depth=max_depth,
                                     gamma=gamma, subsample=subsample,colsample_bytree=colsample_bytree)

        else:
            X, Y, X_train, X_test, Y_train, Y_test, Transdata_train, Transdata_test = self.train_split()

            if SMOTE:

                modeling.logistics_mod_xgb_SMOTE(X_train, Y_train, X_test, Y_test, show_result=self.show_result,minority=SMOTE_minority,combined=self.combined,
                                           nthread=nthread, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                                           gamma=gamma,subsample=subsample, colsample_bytree=colsample_bytree)

            else:

                modeling.logistics_mod_xgb(X_train, Y_train, X_test, Y_test, show_result=self.show_result,nthread=nthread, learning_rate=learning_rate, n_estimators=n_estimators,max_depth=max_depth,
                                     gamma=gamma, subsample=subsample,colsample_bytree=colsample_bytree)

    def NN(self,intermediate_dim,SMOTE=False,SMOTE_minority=0, loss = 'binary_crossentropy',epochs = 20,batch_size=256,dropout = 0.4,activation='relu',output_activation =  'sigmoid'):

        WOE = self.WOE

        if WOE:
            X_woe, Y_woe, X_woe_train, X_woe_test, Y_woe_train, Y_woe_test, Transdata_woe_train, Transdata_woe_test = self.train_split()

            if SMOTE:

                modeling.NN_woe_SMOTE_mod(X_woe_train,Y_woe_train, X_woe_test,Y_woe_test, loss=loss,minority=SMOTE_minority,combined=self.combined,
                                          intermediate_dim=intermediate_dim,epochs=epochs,batch_size=batch_size,dropout = dropout,activation = activation,output_activation = output_activation,show_result=self.show_result)

            else:

                modeling.NN_woe_mod(X_woe_train, Y_woe_train, X_woe_test, Y_woe_test, loss=loss,intermediate_dim=intermediate_dim, epochs=epochs, batch_size=batch_size,
                                          dropout=dropout, activation=activation, output_activation = output_activation,show_result=self.show_result)

        else:
            X, Y, X_train, X_test, Y_train, Y_test, Transdata_train, Transdata_test = self.train_split()

            if SMOTE:

                modeling.NN_SMOTE_mod(X_train, Y_train, X_test, Y_test, loss=loss,
                                          minority=SMOTE_minority, combined=self.combined,
                                          intermediate_dim=intermediate_dim, epochs=epochs, batch_size=batch_size,
                                          dropout=dropout, activation=activation, output_activation = output_activation,show_result=self.show_result)

            else:

                modeling.NN_mod(X_train, Y_train, X_test, Y_test, loss=loss,
                                          intermediate_dim=intermediate_dim, epochs=epochs, batch_size=batch_size,
                                          dropout=dropout, activation=activation,output_activation = output_activation,show_result=self.show_result)

if __name__ == '__main__':

    print('-------------------------------Data Processing-----------------------------')
    Transdata = Telephone_Fraud_Data_Processor().data_processing(X=pd.read_table(path['X_path'], ','),Y=pd.read_table(path['Y_path'],','),
                                                                 worktime=worktime, NewPhone=Newphone, least_time=least_time, combined=combined)

    Data_Modeling = Telephone_Fraud_Data_Modeling(data=Transdata, WOE=False,combined=combined,dummy_feature=dummy_feature,drop_feature=drop_feature,
                                                  re_index=re_index, row_name=row_name, show_result=show_result,
                                                  random_state=random_state, test_size=test_size, show_COR=show_COR,show_WOE=show_WOE)

    print('-------------------------------WOE Transformation-----------------------------')
    Transdata_woe = Telephone_Fraud_Data_Processor().WOE_transformation(data=Transdata,methods=methods, group_num=group_num, show_WOE=show_WOE,
                                                                        combined=combined, re_index=re_index, row_name=row_name)

    WOE_Data_Modeling = Telephone_Fraud_Data_Modeling(data=Transdata_woe, WOE=True,combined=combined,dummy_feature=dummy_feature, drop_feature=drop_feature,
                                                  re_index=re_index, row_name=row_name, show_result=show_result,
                                                  random_state=random_state, test_size=test_size, show_COR=show_COR,show_WOE=show_WOE)

    print('-------------------------------1. LR-----------------------------')
    Data_Modeling.LR()

    print('-------------------------------2. LR+SMOTE-----------------------------')
    Data_Modeling.LR(SMOTE=True,SMOTE_minority=minority)

    print('-------------------------------3. LR+WOE-----------------------------')
    WOE_Data_Modeling.LR()

    print('-------------------------------4. LR+WOE+SMOTE-----------------------------')
    WOE_Data_Modeling.LR(SMOTE=True,SMOTE_minority=minority)

    print('-------------------------------5. XGBoost-----------------------------')
    Data_Modeling.XGB()

    print('-------------------------------6. XGBoost + SMOTE-----------------------------')
    Data_Modeling.XGB(SMOTE=True,SMOTE_minority=minority)

    print('-------------------------------7. XGBoost + WOE-----------------------------')
    WOE_Data_Modeling.XGB()

    print('-------------------------------8. XGBoost + WOE + SMOTE-----------------------------')
    WOE_Data_Modeling.XGB(SMOTE=True,SMOTE_minority=minority)

    print('-------------------------------9, XGBoostlr-----------------------------')
    Data_Modeling.XGBLR()

    print('-------------------------------10. XGBoostlr + SMOTE-----------------------------')
    Data_Modeling.XGBLR(SMOTE=True,SMOTE_minority=minority)

    print('-------------------------------11. XGBoostlr + WOE-----------------------------')
    WOE_Data_Modeling.XGBLR()

    print('-------------------------------12. XGBoostlr + WOE +SMOTE-----------------------------')
    WOE_Data_Modeling.XGBLR(SMOTE=True,SMOTE_minority=minority)

    print('-------------------------------13. NN-----------------------------')
    Data_Modeling.NN(intermediate_dim=intermediate_dim)

    print('-------------------------------14. NN+SMOTE-----------------------------')
    Data_Modeling.NN(SMOTE=True,SMOTE_minority=minority,intermediate_dim=intermediate_dim)

    print('-------------------------------15. NN + WOE-----------------------------')
    WOE_Data_Modeling.NN(intermediate_dim=intermediate_dim)

    print('-------------------------------16. NN + WOE + SMOTE-----------------------------')
    WOE_Data_Modeling.NN(SMOTE=True,SMOTE_minority=minority,intermediate_dim=intermediate_dim)

    print('-------------------------------17. SVM-----------------------------')
    Data_Modeling.SVM(grid=False,C=C,kernel=kernel)
    Data_Modeling.SVM(grid=True,grid_C=grid_C,grid_kernel=grid_kernel)

    print('-------------------------------18. SVM+SMOTE-----------------------------')
    Data_Modeling.SVM(grid=False,C=C, kernel=kernel,SMOTE=True,SMOTE_minority=minority)
    Data_Modeling.SVM(grid=True, grid_C=grid_C, grid_kernel=grid_kernel,SMOTE=True,SMOTE_minority=minority)

    print('-------------------------------19. SVM+WOE-----------------------------')
    WOE_Data_Modeling.SVM(grid=False, C=C, kernel=kernel)
    WOE_Data_Modeling.SVM(grid=True, grid_C=grid_C, grid_kernel=grid_kernel)

    print('-------------------------------20. SVM+WOE+SMOTE-----------------------------')
    WOE_Data_Modeling.SVM(grid=False, C=C, kernel=kernel, SMOTE=True, SMOTE_minority=minority)
    WOE_Data_Modeling.SVM(grid=True, grid_C=grid_C, grid_kernel=grid_kernel, SMOTE=True, SMOTE_minority=minority)



