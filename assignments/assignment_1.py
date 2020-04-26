import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lm
import sklearn.metrics as metrics

bottle_df = pd.read_csv("bottle.csv", low_memory=False)

bottle_df.dropna(subset=["Salnty", "STheta", "T_degC"], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    bottle_df[["Salnty", "STheta"]],
    bottle_df["T_degC"], test_size=.2,
    random_state=0)

X_train = X_train.assign(intercept=1)
X_test = X_test.assign(intercept=1)

""" Manual calculation"""
theta_best = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
print("Coefficients: ", theta_best)
y_predict = X_test.dot(theta_best)
y_predict.head()

""" sklearn method """
lm_mod = lm().fit(X_train, y_train)
print('Coefficients: \n', lm_mod.coef_)
y_predict_train_sk = pd.DataFrame(lm_mod.predict(X_train), columns=[
    "y_predict"])
y_predict_test_sk = pd.DataFrame(lm_mod.predict(X_test), columns=[
    "y_predict"])

"""" Evaluate """
print("Model mean squared error: %.2f"
      % metrics.mean_squared_error(y_train, y_predict_train_sk.y_predict))
print("Model explained variance: %.2f"
      % metrics.explained_variance_score(y_train, y_predict_train_sk.y_predict))
print("Model r-squared: %.2f" % metrics.r2_score(y_train,
                                                 y_predict_train_sk.y_predict))

print("Holdout mean squared error: %.2f"
      % metrics.mean_squared_error(y_test, y_predict_test_sk.y_predict))
print("Holdout explained variance: %.2f"
      % metrics.explained_variance_score(y_test, y_predict_test_sk.y_predict))
print("Holdout r-squared: %.2f" % metrics.r2_score(y_test,
                                                   y_predict_test_sk.y_predict))