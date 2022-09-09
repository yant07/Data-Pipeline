import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def read_file_as_dataframe(filename,file_header='infer',file_sep=","):
    '''
    Load the file into a dataframe.
    filename: str, path object.
    col_list: list-like. Return the subset of the columns.
    name_list: array-like. List of columns to use.
    header: int, list of int, none, default 'infer'. Row number(s) to use as the column names, and the start of the data.  
    IFna: if we remain na in dataframe. 'yes' represents the function returns original dataframe, 'no'
          represents the function finally returns the dataframe after dropping na.
    '''
    if not os.path.exists(filename):
        print('File [{}] not exist, please double check!'.format(filename))
    else:
        if os.path.splitext(filename)[1]=='.csv':
            df=pd.read_csv(filename,header=file_header)
        elif os.path.splitext(filename)[1]=='.xlsx':
            df=pd.read_excel(filename,header=file_header)
        elif os.path.splitext(filename)[1]=='.txt':
            df=pd.read_csv(filename,sep=file_sep,header=file_header)
        elif os.path.splitext(filename)[1]=='.json':
            df=pd.read_json(filename)
        return df

class TabularDataPreprocess():
    def __init__(self,df):
        self.df = df
        # self.SentenceList=[]
        # self.WordList_in_SentenceList = []
        # self.WordList = []#one word list
        # self.concat_sen_text = []
    
    def data_cleaning(self):
        df = self.df.copy()

        def del_dup(df):
            df.drop_duplicates(inplace = True)
            return df

        def find_num(df):
            #把数值型数据找出来
            num_df=df.select_dtypes(include=np.number)
            return num_df

        def find_category(df):
            cate_df=df.select_dtypes(include=object)
            return cate_df

        def drop_na(how='any'):
            '''
            how{‘any’, ‘all’}, default ‘any’
            Determine if row or column is removed from DataFrame, when we have at least one NA or all NA.
            ‘any’ : If any NA values are present, drop that row or column.
            ‘all’ : If all values are NA, drop that row or column.
            '''
            df.dropna(how)
            return df

        def MissingData(self, value=None, fill_na_method='mean'):
            '''
            value: scalar, dict, Series, or DataFrame
            Value to use to fill holes (e.g. 0), alternately a dict/Series/DataFrame of values specifying which value to use for each index (for a Series) or column (for a DataFrame). 
            Values not in the dict/Series/DataFrame will not be filled. This value cannot be a list.

            method:{'mean', ‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}
            Method to use for filling holes in reindexed Series pad 
            '''
            #填补缺失值
            if fill_na_method=='mean':
                # 均值填充
                self.df.fillna(data_test.mean())
            else:
                self.df.fillna(method=fill_na_method)
                # ffill: propagate last valid observation forward to next valid backfill 
                # bfill: use next valid observation to fill gap.

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

        def Outlier_threesigma(self,t=3):
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
            new_df=delete_out3sigma(df,t)
            return new_df

        # funcs=[del_dup,find_num,drop_na,Outlier_threesigma]
        # def func_list(data):
        #     for func in funcs:
        #         data=func(data)
        #     return data

        # clean_df = func_list(df)
        return clean_df
    



