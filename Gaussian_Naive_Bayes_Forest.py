


import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

covtype = fetch_covtype()
'''
print(covtype.DESCR)
    =================   ============
    Classes                        7
    Samples total             581012
    Dimensionality                54
    Features                     int
    =================   ============
'''

covtype_X = covtype.data
covtype_y = covtype.target

covtype_X_train, covtype_X_test, covtype_y_train, covtype_y_test = train_test_split(covtype_X, covtype_y, test_size=0.2)

print("전체 데이터 크기: {}".format(covtype_X.shape))
print("학습 데이터 크기: {}".format(covtype_X_train.shape))
print("평가 데이터 크기: {}".format(covtype_X_test.shape))

covtype_df = pd.DataFrame(data=covtype_X)
print("covtype_df.describe: \n{}".format(covtype_df.describe()))

covtype_train_df = pd.DataFrame(data=covtype_X_train)
print("covtype_train_df.describe: \n{}".format(covtype_train_df.describe()))

covtype_test_df = pd.DataFrame(data=covtype_X_test)
print("covtype_test_df.describe: \n{}".format(covtype_test_df.describe()))

scaler = StandardScaler()
covtype_X_train_scale = scaler.fit_transform(covtype_X_train)
covtype_X_test_scale = scaler.transform(covtype_X_test)

covtype_scaler_train_df = pd.DataFrame(data=covtype_X_train_scale)
print("covtype_scaler_train_df.describe: \n{}".format(covtype_scaler_train_df.describe()))

covtype_scaler_test_df = pd.DataFrame(data=covtype_X_test_scale)
print("covtype_scaler_test_df.describe: \n{}".format(covtype_scaler_test_df.describe()))

model = GaussianNB()
model.fit(covtype_X_train_scale, covtype_y_train)

predict = model.predict(covtype_X_train_scale)
acc = metrics.accuracy_score(covtype_y_train, predict)
f1 = metrics.f1_score(covtype_y_train, predict, average=None)

print('Train Accuracy: {}'.format(acc))
print('Train F1 score: {}'.format(f1))

predict = model.predict(covtype_X_test_scale)
acc = metrics.accuracy_score(covtype_y_test, predict)
f1 = metrics.f1_score(covtype_y_test, predict, average=None)

print('test Accuracy: {}'.format(acc))
print('test F1 score: {}'.format(f1))

def make_meshgrid(x, y, h=0.02):
    x_min, x_max = x.min()-1, x.max()+1
    y_min, y_max = y.min()-1, y.min()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    return xx, yy

def plot_contours(clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contour(xx, yy, Z, **params)
    
    return out

X, y = make_blobs(n_samples=1000)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

model = GaussianNB()
model.fit(X, y)

xx, yy = make_meshgrid(X[:, ], X[:, 1])
plot_contours(model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

plt.show()