from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

data = load_iris()
X = data['data']
y = data['target']
feature_names = data['feature_names']
labels = data['target_names']

clf = tree.DecisionTreeClassifier(max_depth=4)
cv = KFold(shuffle=True)

for fold, (train, test) in enumerate(cv.split(X, y)):
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    
    clf.fit(X_train, y_train)
    score = accuracy_score(y_test, clf.predict(X_test))
    print(f'Accuracy: {score}')