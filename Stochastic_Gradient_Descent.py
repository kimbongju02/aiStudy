

from sklearn.linear_model import SGDRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = datasets.fetch_openml('boston', return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = make_pipeline(StandardScaler(), SGDRegressor(loss='squared_error'))
model.fit(X_train, y_train)

print("학습 데이터 점수: {}".format(model.score(X_train, y_train)))
print("평가 데이터 점수: {}".format(model.score(X_test, y_test)))