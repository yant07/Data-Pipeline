import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

def read_file_as_dataframe(filename,file_header='infer',IFna='yes',file_sep=","):
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
        if IFna=='yes':
            return df
        elif IFna=='no':
            return df.dropna()
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

def clean_data_pipe(filename):
    df = read_file_as_dataframe(filename,file_header='infer')
    df.drop_duplicates(inplace = True)
    df = df.select_dtypes(include=np.number)
    df = df.fillna(method='ffill')
    df2=Outlier_zscore(df)
    # df.describe()
    return df2

def transform_data(df):
    df_normed = preprocessing.minmax_scale(df)
    return df_normed

if __name__=='main':
    clean_df = clean_data_pipe('titanic.csv')
    # normed_df = transform_data_pipe(clean_df)
    print(normed_df.head(5))

    titanic = read_file_as_dataframe('titanic.csv')
    X = titanic.drop('survived', axis=1)
    y = titanic['survived']


    numeric_features = ["age", "fare"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["embarked", "sex", "pclass"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf.fit(X_train, y_train)
    print("model score: %.3f" % clf.score(X_test, y_test))