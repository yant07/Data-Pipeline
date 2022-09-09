import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA

def ReadFile(filename,file_sep=",",file_header='infer'):
    import os
    if not os.path.exists(filename):
        print('File [{}] not exist, please double check!'.format(filename))
    else:
        if os.path.splitext(filename)[1]=='.csv':
            df=pd.read_csv(filename)
        elif os.path.splitext(filename)[1]=='.xlsx':
            df=pd.read_excel(filename)
        elif os.path.splitext(filename)[1]=='.txt':
            df=pd.read_csv(filename,sep=file_sep,header=file_header)
        elif os.path.splitext(filename)[1]=='.json':
            df=pd.read_json(filename)
        return df

######DATA CLEANING#######
def __init__(df):
    self.df = df
def Del_dup(self):
    df.drop_duplicates(inplace = True)
    return df
def Find_num(df):
    #把数值型数据找出来
    # df.dtypes
    df= df.select_dtypes("float64")
    return df
def MissingData(df):
    #填补缺失值
    ## 均值填充
    # df= df.fillna(data_test.mean())
    #data_test_filled
    ## 上下值填充
    df = df.fillna(method='ffill')
    # df = df.fillna(method='bfill')
    return df
def Outlier_zscore(df,z_score_threshold=2.2):
    # 通过Z-Score方法判断异常值
    df_zscore = df.copy()  # 复制一个用来存储Z-score得分的数据框
    cols = df.columns  #  获得列表框的列名
    for col in cols:
        df_col = df[col]  #  得到每一列的值
        z_score = (df_col - df_col.mean()) / df_col.std()  #计算每一列的Z-score得分
        df_zscore[col] = z_score.abs() > z_score_threshold  
        # 判断Z-score得分是否大于2.2，如果是则是True，否则为False
        df_drop_outlier = df[df_zscore[col] == False]
    # df_drop_outlier
    # df_zscore
    return df_drop_outlier
def Outlier_threesigma(df,t=3):
    def three_sigma(Ser1,t):
        '''
        Ser1：表示传入DataFrame的某一列。
        '''
        rule = (Ser1.mean()-t*Ser1.std()>Ser1) | (Ser1.mean()+t*Ser1.std()< Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index  #返回落在3sigma之外的行索引值
    def delete_out3sigma(data,t):
        out_index = [] #保存要删除的行索引
        for i in range(data.shape[1]): # 对每一列分别用3sigma原则处理
            index = three_sigma(data.iloc[:,i],t)
            out_index += index.tolist()
        delete_ = list(set(out_index))
        print('所删除的行索引为：',delete_)
        data.drop(delete_,inplace=True)
        return data
    df=delete_out3sigma(df,t)
    return df

DataCleaning_funcs=[Del_dup,Find_num,MissingData,Outlier_zscore]
def clean_pipe(data):
    for func in DataCleaning_funcs:
        data=func(data)
    return data
##########################
####DATA TRANSFORMATION####
def MinMax_Norm(df):
    df_normed = preprocessing.minmax_scale(df)
    return df_normed
def Zscore_Norm(df):
    df_normed = preprocessing.scale(df)
    return df_normed
def Decimal_Norm(df):
    df_normed = df/10**np.ceil(np.log10(df.abs().max()))
    return df_normed

def Binning(col_Index):
    df_binned = pd.qcut(df.iloc[:,col_Index],20,labels=range(20))
    return df_binned

def Bina(data_array,y):#data_array数据类型需为numpy.ndarray
    X=data_array.reshape(-1,1)
    BI=Binarizer(threshold=y).fit_transform(X)
    return BI

def Reduction_pca(df,n='mle'):
    X= preprocessing.scale(df)#标准化处理
    #n_components设置为‘mle’，算法自动选择满足所要求的方差百分比的特征个数；设置为数字n，则对应n个特征
    pca=PCA(n_components=n)
    # pca=PCA(n_components='mle')
    pca.fit(X)
    pca_com=pca.components_
    ratio = pca.explained_variance_ratio_#各成分方差百分比，各变量的方差贡献率
    s=sum(ratio)
    print("前"+str(n)+"个属性解释了数据中"+str(s)+"的变化。")
    return pca_com
def Simple_Sample(df,sam_num):
    df_sample=df.sample(n=sam_num)
    return df_sample
def Label_Encode(df,label_col_index):
    label_col=iris.iloc[:,label_col_index]
    le = preprocessing.LabelEncoder()
    le.fit(label_col)
    encoded_col=le.transform(label_col)
    df.iloc[:,label_col_index]=encoded_col
    return df

DataTransformation_funcs=[Zscore_Norm,Reduction_pca,Simple_Sample]
def transform_pipe(data):
    for func in DataCleaning_funcs:
        data=func(data)
    return data
##########################

######DATA VISUALIZATION#######

##########################
######DATA MODELING#######
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import classification_report
# X, y = iris.data, iris.target
 
x_train = []                        # 存放训练集的属性
y_train = []                        # 存放训练集的标签
x_test = []                         # 存放测试集的属性
y_test = []                         # 存放测试集的标签
                     

'''决策树方法'''
def DecisionTree(testSize = 0.2 ):
    print('------------------------决策树方法------------------------')
    print('原数据集：')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print(clf.score(x_test, y_test))
    print(classification_report(y_test, y_predict))
 
'''随机森林方法'''
def RandomForest(testSize = 0.2 ):
    print('------------------------随机森林方法------------------------')
    print('原数据集：')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_test)
    print(rfc.score(x_test, y_test))
    print(classification_report(y_test, y_predict))
 
'''KNN方法'''
def knn(testSize = 0.2):
    print('------------------------KNN方法------------------------')
    print('原数据集：')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    print(knn.score(x_test, y_test))
    print(classification_report(y_test, y_predict))
 
'''SVC方法'''
def svc(testSize = 0.2 ):
    print('------------------------SVC方法------------------------')
    print('原数据集：')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print(clf.score(x_test, y_test))
    print(classification_report(y_test, y_predict))


##########################