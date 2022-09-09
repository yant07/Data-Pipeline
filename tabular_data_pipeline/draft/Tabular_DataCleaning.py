import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class DataCleaning():

    def __init__(self, filename):
        self.filename= filename

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

    def Del_dup(self,df):
        df.drop_duplicates(inplace = True)
        return df
    
    def Find_num(self,df):
        #把数值型数据找出来
        # df.dtypes
        df= df.select_dtypes("float64")
        return df

    def MissingData(self,df):
        #填补缺失值
        ## 均值填充
        # df= df.fillna(data_test.mean())
        #data_test_filled
        ## 上下值填充
        df = df.fillna(method='ffill')
        # df = df.fillna(method='bfill')
        return df

    def Outlier_zscore(self,df,z_score_threshold=2.2):
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

    def Outlier_threesigma(self,df,t=3):
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

