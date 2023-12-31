


import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.manifold import TSNE
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

wine = load_wine()

wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df['Target'] = wine.target
print("wine_df: \n{}".format(wine_df))

X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

wine_train_df = pd.DataFrame(data=X_train, columns=wine.feature_names)
wine_train_df['Target'] = y_train
print("wine_train_df: \n{}".format(wine_train_df))

wine_test_df = pd.DataFrame(data=X_test, columns=wine.feature_names)
wine_test_df['Target'] = y_test
print("wine_test_df: \n{}".format(wine_test_df))


model = KNeighborsClassifier()
model.fit(X_train, y_train)

print("학습 데이터 점수: {}".format(model.score(X_train, y_train)))
print("평가 데이터 점수: {}".format(model.score(X_test, y_test)))

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

model = KNeighborsClassifier()
model.fit(X_train_scale, y_train)

print("스케일링 후 학습 데이터 점수: {}".format(model.score(X_train_scale, y_train)))
print("스케일링 후 평가 데이터 점수: {}".format(model.score(X_test_scale, y_test)))

estimator = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)

cross_validate(
    estimator=estimator,
    X=X, y=y,
    cv=5,
    n_jobs=multiprocessing.cpu_count(),
    verbose=True
)

pipe = Pipeline(
    [('scaler', StandardScaler()),
     ('model', KNeighborsClassifier())]
)

param_grid = [{'model__n_neighbors': [3, 5, 7],
               'model__weights': ['uniform', 'distance'],
               'model__algorithm': ['ball_tree', 'kd_tree', 'brute']}]

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    n_jobs=multiprocessing.cpu_count(),
    verbose=True
)  
gs.fit(X, y)
gs.best_estimator_
print('GridSearchCV best score: {}'.format(gs.best_score_))

tsne = TSNE(n_components=2)
X_comp = tsne.fit_transform(X)

wine_comp_df = pd.DataFrame(data=X_comp)
wine_comp_df['target'] = y

plt.scatter(X_comp[:, 0], X_comp[:, 1], c=y, 
            cmap=plt.cm.coolwarm, s=20, edgecolors='k')

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

model = KNeighborsClassifier()
model.fit(X_comp, y)
predict = model.predict(X_comp)

xx, yy = make_meshgrid(X_comp[:, 0], X_comp[:, 1])
plot_contours(model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_comp[:, 0], X_comp[:, 1], c=y, 
            cmap=plt.cm.coolwarm, s=20, edgecolors='k')

plt.show()