#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import model_measurement
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def data_processing(X,Y,worktime,NewPhone,least_time,combined = False):

    rawdata_X = X
    rawdata_Y = Y

    Y_raw = np.zeros(len(rawdata_X))
    Y_raw[rawdata_X['主叫号码'].isin(list(rawdata_Y['号码']))]=1
    #合并X和Y
    rawdata=pd.concat([Series(Y_raw),rawdata_X],axis=1)
    rawdata.rename(columns={0:'是否是诈骗电话'}, inplace = True)
    #整理＜0的列
    f = lambda x: x if x>0 else 0
    rawdata["通话总时长"] = rawdata["通话总时长"].apply(f)
    rawdata["平均通话时长"] = rawdata["平均通话时长"].apply(f)
    #筛列
    data = rawdata[rawdata['业务类型']==0]
    #清缺失值
    data = data[data.notnull()]
    #选时段
    #datawork = data[data['小时'].isin(worktime)]
    datawork = data

    #重新索引！！重要！！
    datawork.reset_index(level=None, drop=True,inplace=True, col_level=0, col_fill="")

    #是否是不常用的手机/物联网终端/未知（画直方分布图来判断不常用手机有哪些！！）
    IsOldPhone_raw = np.ones(len(datawork))
    IsOldPhone_raw[datawork['主叫终端品牌'].isin(NewPhone)]=0
    datawork = pd.concat([datawork,Series(IsOldPhone_raw)],axis=1)
    datawork.rename(columns = {0: 'IsOldPhone_raw'}, inplace = True)

    #是否呼叫了银行
    IsBankCall_raw = np.zeros(len(datawork))
    IsBankCall_raw[datawork['特殊呼叫标志']!=0]=1
    datawork = pd.concat([datawork,Series(IsBankCall_raw)],axis=1)
    datawork.rename(columns={0:'IsBankCall_raw'}, inplace = True)

    #是否是在工作时间拨打
    IsWorkhour_raw = np.zeros(len(datawork))
    IsWorkhour_raw[data['小时'].isin(worktime)] = 1
    datawork = pd.concat([datawork, Series(IsWorkhour_raw)], axis=1)
    datawork.rename(columns={0: 'IsWorkhour_raw'}, inplace=True)

    # 是否是在非工作时间拨打
    IsnotWorkhour_raw = np.ones(len(datawork))
    IsnotWorkhour_raw[data['小时'].isin(worktime)] = 0
    datawork = pd.concat([datawork, Series(IsnotWorkhour_raw)], axis=1)
    datawork.rename(columns={0: 'IsnotWorkhour_raw'}, inplace=True)

    #是否归属地未知
    IsPlaceunKnown_raw = np.zeros(len(datawork))
    IsPlaceunKnown_raw[datawork['主叫归属省']=='未知']=1
    datawork = pd.concat([datawork,Series(IsPlaceunKnown_raw)],axis=1)
    datawork.rename(columns={0:'IsPlaceunKnown_raw'}, inplace = True)

    #通话失败次数
    IsFail_raw = np.zeros(len(datawork))
    IsFail_raw[datawork['平均通话时长']<least_time]=1
    datawork = pd.concat([datawork,Series(IsFail_raw)],axis=1)
    datawork.rename(columns={0:'IsFail_raw'}, inplace = True)
    datawork['IsFail_raw'] = datawork['IsFail_raw'] * datawork['呼叫总次数']

    #group by主叫总次数
    from_call_count = datawork['呼叫总次数'].groupby(datawork['主叫号码']).sum()

    #group by平均通话时间
    total_call_time = datawork['通话总时长'].groupby(datawork['主叫号码']).sum()
    duration_mean = total_call_time/from_call_count

    #group by通话失败率
    IsFail_Count = datawork['IsFail_raw'].groupby(datawork['主叫号码']).sum()
    IsFail_Rate = IsFail_Count/from_call_count

    # group by工作非工作时间通话差值
    IsWorkhour_Count = datawork['IsWorkhour_raw'].groupby(datawork['主叫号码']).sum()
    IsnotWorkhour_Count = datawork['IsnotWorkhour_raw'].groupby(datawork['主叫号码']).sum()
    Workhour_Call_diff = IsWorkhour_Count - IsnotWorkhour_Count

    #group by呼叫离散度
    to_call_count = datawork['被叫号码数（去重）'].groupby(datawork['主叫号码']).sum()
    call_diverse = to_call_count/from_call_count

    #group by被叫所属地个数
    def len_unique(x):
        return len(x.unique())
    Called_City_Count = datawork['被叫归属地市'].groupby(datawork['主叫号码']).apply(len_unique)

    #group by不常用的手机使用次数
    IsOldPhone_Count = datawork['IsOldPhone_raw'].groupby(datawork['主叫号码']).sum()

    #group by呼叫银行次数
    IsBankCall_Count = datawork['IsBankCall_raw'].groupby(datawork['主叫号码']).sum()

    #group by是否归属地未知（dummy）
    IsPlaceunKnown = datawork['IsPlaceunKnown_raw'].groupby(datawork['主叫号码']).mean()

    def Var(arr):
        if len(arr) > 1:
            return np.var(arr)
        else:
            return 0
    def Mode(arr):
        counts = np.bincount(arr)
        return np.argmax(counts)
        #Call_time = datawork['小时'].groupby(datawork['主叫号码']).agg(Var)

    #Call_time = datawork['小时'].groupby(datawork['主叫号码']).agg(Mode)

    #group by 因变量！！（dummy）
    Y = datawork['是否是诈骗电话'].groupby(datawork['主叫号码']).mean()

    Transdata = pd.concat([Y, from_call_count, duration_mean, Called_City_Count, IsFail_Rate,
                           call_diverse, IsBankCall_Count, IsOldPhone_Count, IsPlaceunKnown,Workhour_Call_diff], axis=1)
    Transdata.columns = ['是否是诈骗电话', '主叫次数', '平均通话时间', '被叫所属地个数', '通话失败率',
                         '平均呼叫离散率', '呼叫银行次数', '使用旧款手机次数', '归属地是否未知','工作非工作时间通话差值']
    Transdata['是否是诈骗电话'] = Transdata['是否是诈骗电话'].astype(str)
    Transdata['归属地是否未知'] = Transdata['归属地是否未知'].astype(str)
    #清缺失值
    Transdata = Transdata[Transdata.notnull()]
    # 有重复值，要考虑去重
    Transdata = Transdata.drop_duplicates(keep='first')
    #Transdata[Transdata.duplicated()]

    if combined:
        call_cut = np.zeros(len(Transdata['主叫次数']))
        call_cut[Transdata['主叫次数'].values <= 1188.1] = 3
        call_cut[Transdata['主叫次数'].values <= 12.5] = 2
        call_cut[Transdata['主叫次数'].values <= 4.5] = 1
        diverse_cut = np.zeros(len(Transdata['平均呼叫离散率']))
        diverse_cut[Transdata['平均呼叫离散率'].values <= 1.1] = 3
        diverse_cut[Transdata['平均呼叫离散率'].values <= 0.977] = 2
        diverse_cut[Transdata['平均呼叫离散率'].values <= 0.634] = 1
        call_cut = call_cut.astype('object')
        diverse_cut = diverse_cut.astype('object')
        row_name = [row for row in Transdata.index]
        call_diverse_combined = [[a,(b,c)] for a, b, c in zip(row_name, call_cut, diverse_cut)]
        call_diverse_combined = pd.DataFrame(call_diverse_combined)
        call_diverse_combined.columns = ['主叫号码', '主叫_离散率组合']
        call_diverse_combined.index = call_diverse_combined['主叫号码']
        call_diverse_combined = call_diverse_combined['主叫_离散率组合']
        Transdata = pd.concat([Transdata, call_diverse_combined], axis=1)
        #dummy_call_diverse_combined = pd.get_dummies(call_diverse_combined, prefix='主叫_离散率组合')
        #Transdata = pd.concat([Transdata, call_diverse_combined,dummy_call_diverse_combined.astype(float)], axis=1)

    else:
        pass

    print(Transdata.info())
    #print(Transdata.groupby(['是否是诈骗电话']).count())
    #Transdata.hist()

    return Transdata

def data_dummy(Transdata,dummy_feature):
    for i in range(len(dummy_feature)):
        dummy = pd.get_dummies(Transdata['{}'.format(dummy_feature[i])],prefix=dummy_feature[i])
        Transdata = pd.concat([Transdata, dummy], axis=1)
        Transdata = Transdata.drop(['{}'.format(dummy_feature[i])],axis=1)

    # print(Transdata.info())
    # print(Transdata.groupby(['是否是诈骗电话']).count())
    return Transdata

def train_split(data,random_state,test_size,WOE,combined,dummy_feature):
    if WOE:
        X, Y = data.iloc[:, 1:], data.iloc[:, 0]
    else:
        if combined:
            Transdata_dummy = data_dummy(data, dummy_feature=dummy_feature)
            X, Y = Transdata_dummy.iloc[:, 1:], data.iloc[:, 0]
        else:
            if dummy_feature==None:
                X, Y = data.iloc[:, 1:], data.iloc[:, 0]
            else:
                Transdata_dummy = data_dummy(data, dummy_feature=dummy_feature)
                X, Y = Transdata_dummy.iloc[:, 1:], data.iloc[:, 0]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    Transdata_train = pd.concat([Y_train, X_train], axis=1)
    Transdata_test = pd.concat([Y_test, X_test], axis=1)
    #print(Transdata_test.info())
    #print(Transdata_test.groupby(['是否是诈骗电话']).count())

    return X,Y,X_train,X_test,Y_train ,Y_test,Transdata_train,Transdata_test

def data_SMOTE(X_train,Y_train,minority,combined = False,WOE = False):

    #Transdata_train_SMOTE = pd.concat([Y_train, X_train], axis=1)

    sm = SMOTE(ratio={'1.0': minority}, random_state=42)
    X_train_SMOTE, Y_train_SMOTE = sm.fit_resample(X_train, Y_train)

    Transdata_train_SMOTE = pd.DataFrame()
    Transdata_train_SMOTE['是否是诈骗电话'] = Y_train_SMOTE
    Transdata_train_SMOTE['主叫次数'] = X_train_SMOTE[:, 0]
    Transdata_train_SMOTE['平均通话时间'] = X_train_SMOTE[:, 1]
    Transdata_train_SMOTE['被叫所属地个数'] = X_train_SMOTE[:, 2]
    Transdata_train_SMOTE['通话失败率'] = X_train_SMOTE[:, 3]
    Transdata_train_SMOTE['平均呼叫离散率'] = X_train_SMOTE[:, 4]
    Transdata_train_SMOTE['呼叫银行次数'] = X_train_SMOTE[:, 5]
    Transdata_train_SMOTE['使用旧款手机次数'] = X_train_SMOTE[:, 6]
    Transdata_train_SMOTE['工作非工作时间通话差值'] = X_train_SMOTE[:, 7]

    if combined:
        if WOE:
            Transdata_train_SMOTE['归属地是否未知'] = X_train_SMOTE[:, 8]
            Transdata_train_SMOTE['主叫_离散率组合'] = X_train_SMOTE[:, 9]
        else:
            Transdata_train_SMOTE['主叫_离散率组合_(1.0, 1.0)'] = X_train_SMOTE[:, 8]
            Transdata_train_SMOTE['主叫_离散率组合_(1.0, 2.0)'] = X_train_SMOTE[:, 9]
            Transdata_train_SMOTE['主叫_离散率组合_(1.0, 3.0)'] = X_train_SMOTE[:, 10]
            Transdata_train_SMOTE['主叫_离散率组合_(2.0, 1.0)'] = X_train_SMOTE[:, 11]
            Transdata_train_SMOTE['主叫_离散率组合_(2.0, 2.0)'] = X_train_SMOTE[:, 12]
            Transdata_train_SMOTE['主叫_离散率组合_(2.0, 3.0)'] = X_train_SMOTE[:, 13]
            Transdata_train_SMOTE['主叫_离散率组合_(3.0, 1.0)'] = X_train_SMOTE[:, 14]
            Transdata_train_SMOTE['主叫_离散率组合_(3.0, 2.0)'] = X_train_SMOTE[:, 15]
            Transdata_train_SMOTE['主叫_离散率组合_(3.0, 3.0)'] = X_train_SMOTE[:, 16]
            Transdata_train_SMOTE['归属地是否未知_0.0'] = X_train_SMOTE[:, 17]
            Transdata_train_SMOTE['归属地是否未知_1.0'] = X_train_SMOTE[:, 18]
    else:
        Transdata_train_SMOTE['归属地是否未知'] = X_train_SMOTE[:, 8]

    return X_train_SMOTE, Y_train_SMOTE, Transdata_train_SMOTE