import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier

# we create 40 separable points
rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
"""
randn Return a sample (or samples) from the “standard normal” distribution.
n_samples_1 and 2 mean dimension, n_samples_1 rows and 2 columns
"""
X = np.r_[1.5 * rng.randn(n_samples_1, 2),
          0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)
"""""
np.linspace:
Returns num evenly spaced samples, calculated over the interval [start, stop ]
the number of sample is default 50
""""
xx = np.linspace(-5, 5)


"""""
Replacing SVC(kernel="linear") with SGDClassifier(loss="hinge") only works
When parameters alpha in SGDClassifier is 0.01 while the default is 0.0001
"""""
# fit the model and get the separating hyperplane
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)
w = clf.coef_[0]
a = -w[0] / w[1]
yy = a * xx - clf.intercept_[0] / w[1]


clf_SGD = SGDClassifier(n_iter=100, alpha=0.01, loss="hinge")
clf_SGD.fit(X, y)
w_SGD = clf_SGD.coef_[0]
a_SGD = -w_SGD[0] / w_SGD[1]
yy_SGD = a_SGD * xx - clf_SGD.intercept_[0] / w_SGD[1]


##get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]


# plot separating hyperplanes and samples
h0 = plt.plot(xx, yy, 'k-', label='no weights')
h3 = plt.plot(xx, yy_SGD, 'r-', label='no weights_SGD')
h2 = plt.plot(xx, wyy, 'k--', label='with weights_Class_weight')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.legend()

plt.axis('tight')
plt.show()
