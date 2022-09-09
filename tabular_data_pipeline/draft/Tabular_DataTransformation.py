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