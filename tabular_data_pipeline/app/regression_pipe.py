from sklearn.linear_model import LinearRegression
import utils
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def regression_pipe(file_name):
    anscombe = utils.read_file_as_dataframe(file_name)
    anscombe_1 = anscombe.loc[anscombe['dataset']=='I']

    X_4=[]
    for x in anscombe['x']:
        X_4.append([x])
    y_4=[]
    for y in anscombe['y']:
        y_4.append([y])

    X_train, X_test, y_train, y_test = train_test_split(X_4, y_4, test_size=0.2, random_state=0)
    X_train = np.array(X_train).reshape(-1, 1)
    clf_regression = LinearRegression()
    clf_regression.fit(X_train, y_train)
    print("linear regression score: %.3f" % clf_regression.score(X_test, y_test))

    scores = cross_val_score(clf_regression, X_4, y_4, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    # Show the results of a linear regression within each dataset
    sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=anscombe,
            col_wrap=2, ci=None, palette="muted", height=4,
            scatter_kws={"s": 50, "alpha": 1})

if __name__ == '__main__':
    regression_pipe('c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\datapipline_outline\\Data-Pipeline\\app\\anscombe.csv')
    print("regression_pipe has been created!")