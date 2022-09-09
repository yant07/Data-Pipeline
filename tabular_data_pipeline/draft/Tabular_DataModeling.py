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
testSize = 0.2                      # 测试数据所占总数据比重

'''决策树方法'''
def DecisionTree(testSize):
    print('------------------------决策树方法------------------------')
    print('原数据集：')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print(clf.score(x_test, y_test))
    print(classification_report(y_test, y_predict))
 
'''随机森林方法'''
def RandomForest(testSize):
    print('------------------------随机森林方法------------------------')
    print('原数据集：')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_test)
    print(rfc.score(x_test, y_test))
    print(classification_report(y_test, y_predict))
 
'''KNN方法'''
def knn(testSize):
    print('------------------------KNN方法------------------------')
    print('原数据集：')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    print(knn.score(x_test, y_test))
    print(classification_report(y_test, y_predict))
 
'''SVC方法'''
def svc(testSize):
    print('------------------------SVC方法------------------------')
    print('原数据集：')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print(clf.score(x_test, y_test))
    print(classification_report(y_test, y_predict))