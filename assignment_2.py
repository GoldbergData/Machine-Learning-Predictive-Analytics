import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lm
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing

bottle_df = pd.read_csv("bottle.csv", low_memory=False)

bottle_df.dropna(subset=["Salnty", "STheta", "T_degC"], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    bottle_df[["Salnty", "STheta"]],
    bottle_df["T_degC"], test_size=.20,
    random_state=0)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
y_train = np.asarray(y_train).reshape((len(y_train), 1))

X_test = scaler.transform(X_test)
y_test = np.asarray(y_test).reshape((len(y_test), 1))

np.concatenate(np.ones((2, 1)), X_train, axis=1)
np.c_[np.ones((2, 1)), X_test]

X_train = X_train.assign(intercept=1.0)
X_test = X_test.assign(intercept=1.0)

theta_path_mgd = []
best_thetas = []
m = len(X_train)
eta = 0.1
n_iterations = 100
minibatch_size = [50, 2000, 10000]

np.random.seed(42)
theta = np.random.randn(3, 1)  # random initialization

for size in minibatch_size:
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_train[shuffled_indices]
        y_shuffled = y_train[shuffled_indices]
        for i in range(0, m, size):
            xi = X_b_shuffled[i:i + size]
            yi = y_shuffled[i:i + size]
            gradients = 2 / size * np.asarray(xi).T.dot(xi.dot(theta) - yi)
            theta = theta - eta * gradients
            theta_path_mgd.append(theta)
    best_thetas.append(theta)

(lm().fit(X_train, y_train)).coef_

y_predict_50 = X_test.dot(best_thetas[0])
y_predict_2000 = X_test.dot(best_thetas[1])
y_predict_10000 = X_test.dot(best_thetas[2])

print("Holdout mean squared error: %.2f"
      % metrics.mean_squared_error(y_test, y_predict_10000))
print("Holdout explained variance: %.2f"
      % metrics.explained_variance_score(y_test, y_predict_10000))
print("Holdout r-squared: %.2f" % metrics.r2_score(y_test,
                                                   y_predict_10000))

for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_train[shuffled_indices]
    y_shuffled = y_train[shuffled_indices]
    for i in range(0, m, minibatch_size):
        xi = X_b_shuffled[i:i + minibatch_size]
        yi = y_shuffled[i:i + minibatch_size]
        gradients = 2 / minibatch_size * np.asarray(xi).T.dot(xi.dot(theta) -
                                                              yi)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)


