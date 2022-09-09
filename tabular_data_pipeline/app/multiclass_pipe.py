import utils
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import make_column_selector
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
def iris_multiclass_pipe():
    iris = utils.read_file_as_dataframe('c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\datapipline_outline\\Data-Pipeline\\app\\iris.csv')

    preprocessor_iris = ColumnTransformer(
        transformers=[
        ('scale', StandardScaler(),
        make_column_selector(dtype_include=np.number)),
        ('onehot',
        OneHotEncoder(),
        make_column_selector(dtype_include=object))])
    # preprocessor_iris.fit_transform(iris)

    clf_svc = Pipeline(
        [
            ("preprocessor_iris", preprocessor_iris),
            ("svc", SVC(gamma="auto")),
        ]
    )
    clf_decision_tree = Pipeline(
        [
            ("preprocessor_iris", preprocessor_iris),
            ("decision_tree", tree.DecisionTreeClassifier()),
        ]
    )
    clf_random_forest = Pipeline(
        [
            ("preprocessor_iris", preprocessor_iris),
            ("random_forest", RandomForestClassifier(max_depth=2, random_state=0)),
        ]
    )
    clf_gbdt = Pipeline(
        [
            ("preprocessor_iris", preprocessor_iris),
            ("GBDT", GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)),
        ]
    )

    X_3 = iris.drop('species', axis=1)
    y_3 = iris['species']
    X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.4, random_state=0)

    clf_svc.fit(X_train, y_train)
    print("svm model score: %.3f" % clf_svc.score(X_test, y_test))
    clf_decision_tree.fit(X_train, y_train)
    print("decision tree model score: %.3f" % clf_decision_tree.score(X_test, y_test))
    clf_random_forest.fit(X_train, y_train)
    print("random forest model score: %.3f" % clf_random_forest.score(X_test, y_test))
    clf_gbdt.fit(X_train, y_train)
    print("gradient boosting model score: %.3f" % clf_gbdt.score(X_test, y_test))

    sns.scatterplot(data=iris,x="petal_width",y="petal_length")
    plt.show()
    sns.histplot(data=iris["sepal_length"])
    plt.show()
    # sns.distplot(iris["sepal_length"])
    iris.iloc[:,0:4].boxplot()
    plt.show()
    sns.pairplot(iris,hue="species")
    plt.show()


if __name__ == '__main__':
    iris_multiclass_pipe()
    print("multiclass pipe has been created!")