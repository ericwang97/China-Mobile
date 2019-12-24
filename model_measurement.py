# 绘制ROC曲线
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd #数据分析
import numpy as np #科学计算
from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def VIF(data,name,combined = False,WOE = False):

    vif = pd.DataFrame()

    if combined:
        vif["features"] = data.columns
        vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    else:
        vif["features"] = data.columns
        if WOE:
            vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        else:
            if '归属地是否未知'in data.columns:
                data['归属地是否未知'] = data['归属地是否未知'].astype(float)
                vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
                data['归属地是否未知'] = data['归属地是否未知'].astype(str)
            else:
                vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    print('VIF of {}'.format(name))
    print(vif)
    return vif

def plotROC(predictions, Y_test, name, method = 'normal',show = True):
    if method == 'normal':
        fpr, tpr, thresholds = roc_curve(np.array(Y_test).astype(float), predictions[:, 1])
    else:
        fpr, tpr, thresholds = roc_curve(np.array(Y_test).astype(float), predictions.astype(float))
    roc_auc = auc(fpr, tpr)
    print('AUC of {} is: {:0.3f} '.format(name, roc_auc))
    if show:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.grid()
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.title('AUC of {} is: {:0.3f} '.format(name, roc_auc))
        plt.show()
    else:
        pass

def coef(model,feature,Y_train):
    coef = pd.DataFrame()
    coef["features"] = feature.columns
    coef["coef"] = model.coef_[0]
    print(coef)

    X_train1 = sm.add_constant(feature)
    logit = sm.Logit(Y_train.astype(float), X_train1.astype(float), solver='liblinear')
    result = logit.fit()
    print(result.summary())

# 绘制KS曲线
def plotKS(predictions, Y_test, name, method = 'normal'):
    if method == 'normal':
        fpr, tpr, thresholds = roc_curve(np.array(Y_test).astype(float), predictions[:, 1])
    else:
        fpr, tpr, thresholds = roc_curve(Y_test, predictions)

    ks_value = max(tpr - fpr)

    plt.plot(1 - thresholds, tpr, label='tpr')  # ks曲线要按照预测概率降序排列，所以需要1-threshold镜像
    plt.plot(1 - thresholds, fpr, label='fpr')
    plt.plot(1 - thresholds, tpr - fpr, label='KS')
    plt.xlabel('score')
    plt.grid()
    plt.title('KS value of {} is: {:0.3f} '.format(name, ks_value))
    plt.ylim([0.0, 1.0])
    plt.xlim([0.95, 1.0])
    plt.legend(loc='upper left')
    plt.show()

#绘制相关系数矩阵

def plotCOR(Transdata,WOE = False):

    if WOE:
        title = 'COR Matrix of WOE data '
        corr = Transdata.corr()

    else:
        title = 'COR Matrix of data '
        if '是否是诈骗电话' in Transdata.columns and '归属地是否未知' in Transdata.columns:
            Transdata['是否是诈骗电话'] = Transdata['是否是诈骗电话'].astype(float)
            Transdata['归属地是否未知'] = Transdata['归属地是否未知'].astype(float)
            corr = Transdata.corr()
            Transdata['是否是诈骗电话'] = Transdata['是否是诈骗电话'].astype(str)
            Transdata['归属地是否未知'] = Transdata['归属地是否未知'].astype(str)
        if '是否是诈骗电话' in Transdata.columns and '归属地是否未知' not in Transdata.columns:
            Transdata['是否是诈骗电话'] = Transdata['是否是诈骗电话'].astype(float)
            corr = Transdata.corr()
            Transdata['是否是诈骗电话'] = Transdata['是否是诈骗电话'].astype(str)
        if '是否是诈骗电话' not in Transdata.columns and '归属地是否未知' in Transdata.columns:
            Transdata['归属地是否未知'] = Transdata['归属地是否未知'].astype(float)
            corr = Transdata.corr()
            Transdata['归属地是否未知'] = Transdata['归属地是否未知'].astype(str)
        if '是否是诈骗电话' not in Transdata.columns and '归属地是否未知' not in Transdata.columns:
            corr = Transdata.corr()
            pass

    # 数据相关系数的一半
    mask = np.array(corr)
    mask[np.tril_indices_from(mask)] = False
    fig = plt.figure()
    # 建立画纸
    ax1 = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(15, 15)
    # 使用heatmap胜利图
    sns.heatmap(corr, mask = mask,annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
    plt.xticks(fontsize=10, color='black', rotation=45)
    plt.yticks(fontsize=10, color='black', rotation=45)
    plt.title(title)
    plt.show()

#画Learning Curve判断是否过欠拟合
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(0.1, 1., 30), verbose=0, plot=True):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=4, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)    #CV: The default value will change from 3 to 5

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:

        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"number of observations")
        plt.ylabel(u"score")
        #plt.gca().invert_yaxis()
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"Scores on the training data")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"Scores on the CV data")
        plt.legend(loc="best")

        plt.draw()
        plt.show()