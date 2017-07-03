"""
GridSearchCV can be used for other estimators too, not just for
support vectors machine, check out example on the bottom of this code
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()

# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, iris.data, iris.target, cv=5)

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 0.00001],
                     'C': [1, 10, 100, 1000, 10000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

clf = GridSearchCV(SVC, tuned_parameters, cv=5).fit(X_train, y_train)

clf.best_params_
clf.cv_results_['mean_test_score']
clf.cv_results_['std_test_score']


"""
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)


rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)

param_grid = {
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
print CV_rfc.best_params_
"""
