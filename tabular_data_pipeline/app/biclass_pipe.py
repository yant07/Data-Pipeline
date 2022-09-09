import utils
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

def biclass_pipe():
    titanic = utils.read_file_as_dataframe('c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\datapipline_outline\\Data-Pipeline\\app\\titanic.csv')
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
    sns.barplot(data=titanic,x='class',y='survived')
    sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)

    titanic_pie_data = titanic['sex'].value_counts(ascending=True)
    labels = 'female','male'
    fig1, ax1 = plt.subplots()
    ax1.pie(titanic_pie_data, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

    plt.show()


if __name__ == '__main__':
    biclass_pipe()
    print("biclass pipe has been created!")