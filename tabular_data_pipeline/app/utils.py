import os
import pandas as pd
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