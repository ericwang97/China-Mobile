import matplotlib.pyplot as plt
import pandas as pd #数据分析
import numpy as np #科学计算
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def WOE_Binning(data, feature, group_num=0, method='Equval_length', cut_or_qcut=True, cut_manual=[]):

    data['{}'.format(feature)] = data['{}'.format(feature)].astype(float)

    if method == 'Equval_length':
        group_cut = pd.cut(data['{}'.format(feature)], group_num, include_lowest=True)

    if method == 'Equal_frequency':
        try:
            group_cut = pd.qcut(data['{}'.format(feature)], group_num)
        except:
            group_cut = pd.qcut(data['{}'.format(feature)], group_num, duplicates='drop', precision=3)

    if method == 'Optimal':  # 分箱的核心是用机器来选最优的分箱节点
        r = 0
        x = data['{}'.format(feature)].values
        y = data['是否是诈骗电话'].values
        group_max = group_num
        while np.abs(r) < 1:
            d1 = pd.DataFrame(
                {"X": x, "Y": y, "Boundary": pd.qcut(x, group_max)})  # 注意这里是pd.qcut, Bucket：将 X 分为 n 段，n由斯皮尔曼系数决定
            d2 = d1.groupby('Boundary', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)  # 以斯皮尔曼系数作为分箱终止条件
            group_max = group_max - 1
        try:
            group_cut = pd.qcut(data['{}'.format(feature)], group_max)
        except:
            group_cut = pd.qcut(data['{}'.format(feature)], group_max, duplicates='drop', precision=3)

    if method == 'Kmeans':
        kmodel = KMeans(n_clusters=group_num, n_jobs=4)  # 建立模型，n_jobs是并行数，一般等于CPU个数
        kmodel.fit(data['{}'.format(feature)].values.reshape((len(data['{}'.format(feature)]), 1)))
        # 训练模型，reshape是数组的方法
        cut1 = pd.DataFrame(kmodel.cluster_centers_).sort_values(by=0)  # 输出聚类中心，并且排序
        cut2 = cut1.rolling(2).mean().iloc[1:]  # 相邻两项求中点，作为边界点
        cut3 = [0] + list(cut2[0]) + [data['{}'.format(feature)].max()]  # 把首末边界点加上
        group_cut = pd.cut(data['{}'.format(feature)], cut3, include_lowest=True)

    if method == 'Tree':
        boundary = []
        x = data['{}'.format(feature)].values
        y = data['是否是诈骗电话'].values
        clf = DecisionTreeClassifier(criterion='gini',  # “信息熵”最小化准则划分
                                     max_leaf_nodes=group_num,  # 最大叶子节点数
                                     min_samples_leaf=0.05)  # 叶子节点样本数量最小占比
        clf.fit(x.reshape(-1, 1), y)  # 训练决策树

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
                boundary.append(threshold[i])

        boundary.sort()
        min_x = x.min()
        max_x = x.max() + 0.10  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
        boundary = [min_x] + boundary + [max_x]
        group_cut = pd.cut(data['{}'.format(feature)], boundary, include_lowest=True)

    if method == 'Other':
        if cut_or_qcut:
            group_cut = pd.cut(data['{}'.format(feature)], cut_manual, include_lowest=True)
        else:
            try:
                group_cut = pd.qcut(data['{}'.format(feature)], cut_manual)
            except:
                group_cut = pd.qcut(data['{}'.format(feature)], cut_manual, duplicates='drop', precision=3)

    return group_cut

def get_Score(woe, factor):
    scores = []
    for w in woe:
        score = round(w * factor, 0)
        scores.append(score)
    return scores

def get_Table(group_cut, data,re_index = False,row_name = None):
    group_cut_group = data['是否是诈骗电话'].astype(float).groupby(group_cut).count()
    group_cut_group1 = data['是否是诈骗电话'].astype(float).groupby(group_cut).sum()
    cardDf1 = pd.merge(pd.DataFrame(group_cut_group), pd.DataFrame(group_cut_group1), left_index=True, right_index=True)
    cardDfdict = {'是否是诈骗电话_x': '总电话数', '是否是诈骗电话_y': '诈骗电话数'}
    cardDf1.rename(columns=cardDfdict, inplace=True)

    if re_index:
        cardDf1.index = row_name
        cardDf1.insert(2, 'row_name', row_name)
    else:
        pass

    # 计算占比

    cardDf1.insert(2, '诈骗电话占比', cardDf1['诈骗电话数'] / cardDf1['诈骗电话数'].sum())
    cardDf1.insert(2, '分组诈骗电话占比', cardDf1['诈骗电话数'] / cardDf1['总电话数'].sum())
    cardDf1.insert(2, '好电话数', cardDf1['总电话数'] - cardDf1['诈骗电话数'])
    cardDf1.insert(2, '好电话占比', cardDf1['好电话数'] / cardDf1['好电话数'].sum())
    # 计算WOE
    cardDf1.insert(2, 'WOE', np.log(cardDf1['诈骗电话占比']) - np.log(cardDf1['好电话占比']))
    # 计算IV
    cardDf1.insert(2, 'IV', (cardDf1['诈骗电话占比'] - cardDf1['好电话占比']) * cardDf1['WOE'])
    # 计算得分
    basepoint = 600
    PDO = 20
    factor = PDO / np.log(2)
    # offset = basepoint - PDO * np.log(PDO) / np.log(2)
    cardDf1.insert(2, 'Score', get_Score(cardDf1['WOE'], factor))

    return cardDf1

def get_WOE(group_cut, data):
    cardDf1 = get_Table(group_cut, data)

    return cardDf1['WOE']

def get_IV(group_cut, data):
    cardDf1 = get_Table(group_cut, data)
    IV = (cardDf1['IV']).sum()

    return IV

def WOE_Transform(cut,cut_woe):
    a=[]
    for i in cut.unique():
        a.append(i)
        a.sort()
    for m in range(len(a)):
        cut.replace(a[m],cut_woe.values[m],inplace=True)
    return cut

def WOE_Combined_Transform(group_cut,data,name,row_name = None):
    cardDf1 = get_Table(group_cut, data,re_index = True,row_name=row_name)
    table = pd.merge(data['{}'.format(name)], cardDf1.filter(regex='WOE|row_name'), left_on='{}'.format(name),
                      right_on='row_name', how='inner')
    table.index = data.index

    return table['WOE']

def WOE_Show(group_cut, data, feature, output='WOE',re_index = False,row_name = None):
    cardDf1 = get_Table(group_cut, data,re_index = re_index,row_name = row_name)
    IV = (cardDf1['IV']).sum()

    #print('{}特征的IV值为：{:0.3f}'.format(feature, IV))
    print(cardDf1)

    if output == 'WOE':
        # fig,axes=plt.subplots(2,2)
        fig = plt.figure(figsize=(8, 5))
        # 画WOE图
        ax1 = cardDf1['WOE'].plot(linewidth=2, marker='o', title='WOE值随{}的变化趋势图，IV值为{:0.3f}'.format(feature,IV))
        plt.show()

    if output == 'all':
        # 画分布图
        ax1 = cardDf1[["好电话数", "诈骗电话数"]].plot.bar(figsize=(8, 5))
        ax1.set_xticklabels(cardDf1.index, rotation=0)
        ax1.set_ylabel("电话数")
        ax1.set_title("{}与是否是诈骗电话分布图".format(feature))
        plt.show()
        # 画占比图
        fig = plt.figure(figsize=(8, 5))
        ax2 = cardDf1["诈骗电话占比"].plot(linewidth=2, marker='o')
        ax2.set_ylabel("诈骗电话率")
        ax2.set_title("诈骗电话率随{}的变化趋势图".format(feature))
        plt.show()
        # 画WOE图
        fig = plt.figure(figsize=(8, 5))
        ax1 = cardDf1['WOE'].plot(linewidth=2, marker='o', title='WOE值随{}的变化趋势图，IV值为{:0.3f}'.format(feature,IV))
        plt.show()

#   Equval_length,Equal_frequency,Kmeans,Optimal,Tree
#  'Other',cut_manual = [0,0.02,0.15,0.2,0.5,1]
#  ['主叫次数', '平均通话时间', '被叫所属地个数', '通话失败率','工作时间通话次数', '平均呼叫离散率', '呼叫银行次数', '使用旧款手机次数', '归属地是否未知']工作非工作时间通话差值，主叫_离散率组合
#  IV至少要＞0.02才是有用的信息（类似于相关系数初筛）；IV＞0.1的要有5个以上，防止欠拟合

def WOE_main_func(data,methods,group_num,show_WOE = False,combined = False,re_index = False,row_name = None):

    cut_1 = WOE_Binning(data,[column for column in data][1:][0],group_num[0],method = methods[0])         #主叫次数
    cut_IV_1 = get_IV(cut_1,data)
    cut_woe_1 = get_WOE(cut_1,data)

    cut_2 = WOE_Binning(data,[column for column in data][1:][1],group_num[1],method = methods[1])         #平均通话时间，不单调，解释性还可以
    cut_IV_2 = get_IV(cut_2,data)
    cut_woe_2 = get_WOE(cut_2,data)

    cut_3 = WOE_Binning(data,[column for column in data][1:][2],group_num[2],method = methods[2])         #被叫所属地个数,解释性还可以
    cut_IV_3 = get_IV(cut_3,data)
    cut_woe_3 = get_WOE(cut_3,data)

    #cut_4 = WOE_Binning(Transdata,[column for column in Transdata][1:][3],group_num[3],method = methods[3])         #通话失败率，手动调参合并分组
    cut_4 = WOE_Binning(data, [column for column in data][1:][3], group_num[3], method=methods[3],
                        cut_or_qcut=True, cut_manual=[0, 0.0762, 0.559, 1])
    cut_IV_4 = get_IV(cut_4,data)
    cut_woe_4 = get_WOE(cut_4,data)

    #cut_5 = WOE_Binning(Transdata,[column for column in Transdata][1:][4],group_num[4],method = methods[4])         #平均呼叫离散率，目前可以调成2强行单调，对结果影响不大
    cut_5 = WOE_Binning(data, [column for column in data][1:][4], group_num[4], method=methods[4])
    cut_IV_5 = get_IV(cut_5,data)
    cut_woe_5 = get_WOE(cut_5,data)

    cut_6 = WOE_Binning(data,[column for column in data][1:][5],group_num[5],
                        method = methods[5],cut_or_qcut = True,cut_manual = [0,0.2,0.4,1])                          #呼叫银行次数，这两个没救了
    cut_IV_6 = get_IV(cut_6,data)
    cut_woe_6 = get_WOE(cut_6,data)

    cut_7 = WOE_Binning(data,[column for column in data][1:][6],group_num[6],method = methods[6])         #使用旧款手机次数，这两个没救了
    cut_IV_7 = get_IV(cut_7,data)
    cut_woe_7 = get_WOE(cut_7,data)

    cut_8 = WOE_Binning(data,[column for column in data][1:][7],group_num[7],method = methods[7])       #归属地是否未知，目前只能分2组，Tree和Kmeans结果一样。等新数据
    cut_IV_8 = get_IV(cut_8,data)
    cut_woe_8 = get_WOE(cut_8,data)

    cut_9 = WOE_Binning(data, [column for column in data][1:][8],group_num[8],method = methods[8])       #新增变量！！看起来还可以，工作时间通话次数
    cut_IV_9 = get_IV(cut_9, data)
    cut_woe_9 = get_WOE(cut_9, data)

    if combined:

        cut_IV_10 = get_IV([cut_1, cut_5], data)
        #cut_woe_10 = get_WOE([cut_1, cut_5], Transdata)

    else:
        pass

    if combined:
        IV = pd.DataFrame(
            [cut_IV_1, cut_IV_2, cut_IV_3, cut_IV_4, cut_IV_5, cut_IV_6, cut_IV_7, cut_IV_8, cut_IV_9, cut_IV_10],
            index=['主叫次数', '平均通话时间', '被叫所属地个数', '通话失败率', '平均呼叫离散率',
                   '呼叫银行次数', '使用旧款手机次数', '归属地是否未知', '工作非工作时间通话差值', '主叫_离散率组合'], columns=['IV'])
        print(IV['IV'].sort_values(ascending=False))

        if show_WOE:
            WOE_Show(cut_1, data, '主叫次数', 'WOE')
            WOE_Show(cut_2, data, '平均通话时间', 'WOE')
            WOE_Show(cut_3, data, '被叫所属地个数', 'WOE')
            WOE_Show(cut_4, data, '通话失败率', 'WOE')
            WOE_Show(cut_5, data, '平均呼叫离散率', 'WOE')
            WOE_Show(cut_6, data, '呼叫银行次数', 'WOE')
            WOE_Show(cut_7, data, '使用旧款手机次数', 'WOE')
            WOE_Show(cut_8, data, '归属地是否未知', 'WOE')
            WOE_Show(cut_9, data, '工作非工作时间通话差值', 'WOE')
            WOE_Show([cut_1,cut_5], data, '主叫_离散率组合', 'WOE',re_index = re_index,row_name=row_name)

            iv = IV['IV'].sort_values(ascending=False).plot.bar(color='b', rot=30, figsize=(10, 5), fontsize=(10))
            iv.set_title('特征变量与IV值分布图', fontsize=(15))
            iv.set_xlabel('特征变量', fontsize=(15))
            iv.set_ylabel('IV', fontsize=(15))
            plt.show()

        else:
            pass

    else:
        IV = pd.DataFrame(
            [cut_IV_1, cut_IV_2, cut_IV_3, cut_IV_4, cut_IV_5, cut_IV_6, cut_IV_7, cut_IV_8, cut_IV_9],
            index=['主叫次数', '平均通话时间', '被叫所属地个数', '通话失败率', '平均呼叫离散率',
                   '呼叫银行次数', '使用旧款手机次数', '归属地是否未知', '工作非工作时间通话差值'], columns=['IV'])
        print(IV['IV'].sort_values(ascending=False))

        if show_WOE:
            WOE_Show(cut_1, data, '主叫次数', 'WOE')
            WOE_Show(cut_2, data, '平均通话时间', 'WOE')
            WOE_Show(cut_3, data, '被叫所属地个数', 'WOE')
            WOE_Show(cut_4, data, '通话失败率', 'WOE')
            WOE_Show(cut_5, data, '平均呼叫离散率', 'WOE')
            WOE_Show(cut_6, data, '呼叫银行次数', 'WOE')
            WOE_Show(cut_7, data, '使用旧款手机次数', 'WOE')
            WOE_Show(cut_8, data, '归属地是否未知', 'WOE')
            WOE_Show(cut_9, data, '工作非工作时间通话差值', 'WOE')

            iv = IV['IV'].sort_values(ascending=False).plot.bar(color='b', rot=30, figsize=(10, 5), fontsize=(10))
            iv.set_title('特征变量与IV值分布图', fontsize=(15))
            iv.set_xlabel('特征变量', fontsize=(15))
            iv.set_ylabel('IV', fontsize=(15))
            plt.show()

        else:
            pass


    Transdata_woe=pd.DataFrame()   #新建df_new存放woe转换后的数据

    Transdata_woe['是否是诈骗电话']=data["是否是诈骗电话"]
    Transdata_woe['主叫次数']=WOE_Transform(cut_1,cut_woe_1)
    Transdata_woe['平均通话时间']=WOE_Transform(cut_2,cut_woe_2)
    Transdata_woe['被叫所属地个数']=WOE_Transform(cut_3,cut_woe_3)
    Transdata_woe['通话失败率']=WOE_Transform(cut_4,cut_woe_4)
    Transdata_woe['平均呼叫离散率']=WOE_Transform(cut_5,cut_woe_5)
    Transdata_woe['呼叫银行次数']=WOE_Transform(cut_6,cut_woe_6)
    Transdata_woe['使用旧款手机次数']=WOE_Transform(cut_7,cut_woe_7)
    Transdata_woe['归属地是否未知']=WOE_Transform(cut_8,cut_woe_8)
    Transdata_woe['工作非工作时间通话差值'] = WOE_Transform(cut_9,cut_woe_9)

    if combined:
        Transdata_woe['主叫_离散率组合'] = WOE_Combined_Transform([cut_1, cut_5],data,'主叫_离散率组合',row_name = row_name)
    else:
        pass

    #Transdata_woe.to_csv('WoeData.csv', index=False)

    return Transdata_woe


