


import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['Target'] = iris.target
print("iris_df: \n{}".format(iris_df))

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

iris_train_df = pd.DataFrame(data=X_train, columns=iris.feature_names)
iris_train_df['Target'] = y_train
print("iris_train_df: \n{}".format(iris_train_df))

iris_test_df = pd.DataFrame(data=X_test, columns=iris.feature_names)
iris_test_df['Target'] = y_test
print("iris_test_df: \n{}".format(iris_test_df))

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

print("학습 데이터 점수: {}".format(model.score(X_train, y_train)))
print("평가 데이터 점수: {}".format(model.score(X_test, y_test)))

model = KNeighborsClassifier()
model.fit(X_train_scale, y_train)

print("스케일링 후 학습 데이터 점수: {}".format(model.score(X_train_scale, y_train)))
print("스케일링 후 평가 데이터 점수: {}".format(model.score(X_test_scale, y_test)))

cross_validate(
    estimator=KNeighborsClassifier(),
    X=X, y=y,
    cv=5,
    n_jobs=multiprocessing.cpu_count(),
    verbose=True
)

param_grid = [{'n_neighbors': [3, 5, 7],
               'weights': ['uniform', 'distance'],
               'algorithm': ['ball_tree', 'kd_tree', 'brute']}]

gs = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    n_jobs=multiprocessing.cpu_count(),
    verbose=True
)
gs.fit(X, y)
gs.best_estimator_
print('GridSearchCV best score: {}'.format(gs.best_score_))

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min()-1, x.max()+1
    y_min, y_max = y.min()-1, y.max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contour(xx, yy, Z, **params)
    
    return out

tsne = TSNE(n_components=2)
X_comp = tsne.fit_transform(X)

iris_comp_df = pd.DataFrame(data=X_comp)
iris_comp_df['Target'] = y

plt.scatter(X_comp[:, 0], X_comp[:, 1], c=y, 
            cmap=plt.cm.coolwarm, s=20, edgecolors='k')

model = KNeighborsClassifier()
model.fit(X_comp, y)
predict = model.predict(X_comp)

xx, yy = make_meshgrid(X_comp[:, 0], X_comp[:, 1])
plot_contours(model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_comp[:, 0], X_comp[:, 1], c=y, 
            cmap=plt.cm.coolwarm, s=20, edgecolors='k')

plt.show()