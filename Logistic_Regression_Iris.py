


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from seaborn._core.properties import LineWidth
from matplotlib.colors import ListedColormap
import multiprocessing
from sklearn.model_selection import GridSearchCV

iris = load_iris()
print(iris.keys())
print(iris.DESCR)

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
species = pd.Series(iris.target, dtype='category')
species = species.cat.rename_categories(iris.target_names)
iris_df['species'] = species
print(iris_df.describe())


X_train, X_test, y_train, y_test = train_test_split(iris.data[:, [2, 3]], iris.target,
                                                    test_size=0.2, random_state=1,
                                                    stratify=iris.target)

model = LogisticRegression(solver='lbfgs', multi_class='auto', C=100.0, random_state=1)
model.fit(X_train, y_train)

print("학습 데이터 점수: {}".format(model.score(X_train, y_train)))
print("평가 데이터 점수: {}".format(model.score(X_test, y_test)))

X = np.vstack((X_train, X_test))
y = np.hstack((y_train, y_test))


x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max,0.02),
                       np.arange(x2_min, x2_max, 0.02))
Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)

species = ('Setosa', 'Versicoliur', 'Virginica')
markers = ('^', 'v', 's')
colors = ('blue', 'purple','red')
cmap = ListedColormap(colors[:len(np.unique(y))])
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

for idx, cl in enumerate(np.unique(y)):
  plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
              alpha=0.8, c=colors[idx],
              marker=markers[idx], label=species[cl],
              edgecolor='b')
  
X_comb_test, y_comb_test = X[range(105, 105), :], y[range(105, 105)]
plt.scatter(X_comb_test[:, 0], X_comb_test[:, 1],
            c='yellow', edgecolor='k', alpha=0.2,
            linewidth=1, marker='o',
            s=100, label='Test')

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(loc='upper left')
plt.tight_layout()


param_grid = [{'penalty': ['l1', 'l2'],
               'C': [2.0, 2.2, 2.4, 2.6, 2.8]}]
gs = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid,
                  scoring='accuracy', cv=10, n_jobs=multiprocessing.cpu_count())
#print(gs)


result = gs.fit(iris.data, iris.target)
#print(gs.best_estimator_)
print("최적 점수: {}".format(gs.best_score_))
print("최적 파라미터: {}".format(gs.best_params_))
print(pd.DataFrame(result.cv_results_))



plt.show()