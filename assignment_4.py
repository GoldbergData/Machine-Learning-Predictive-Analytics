"""
Random Forest Binary Classification:

For assignment #4 we will be working with a credit default data set. The data
includes various features around financial history and demographic
information. The target variable is "default payment next week", which is
just a binary flag of whether a customer defaults on a payment in the next
week.

You will need to use the Random Forest Classifier from sklearn in order to
build a classifier to predict if a customer is likely to default. You will
also need to use the GridSearch CV for this assignment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
import warnings
# warnings.filterwarnings("ignore")

""""
1. Data Processing:

a) Import the data: The target / y variable is "default payment next month" 
column. Keep all predictors except for the row column (this is a blank in the 
.xlsx file). 

b) Remove any rows that have missing data.

c) Split data into train / test set using an 70/30 split.

2. Random Forest Classifier - Base Model:

Start by creating a simple Random Forest only using default parameters.

a) Use the RandomForestClassifier in sklearn. Fit your model on the training 
data. 

b) Use the fitted model to predict on test data. Use the .predict_proba() and 
the .predict() methods to get predicted probabilities as well as predicted 
classes. 

c) Calculate the confusion matrix and classification report (both are in 
sklearn.metrics). These are the same tools from HW #3. 

d) Calculate the roc_auc_score for this model. There are many ways to do 
this, but an example is to use the probabilities from step B and utilize the 
roc_auc_score from sklearn. 

Documentation: http://scikit-learn.org/stable/modules/generated/sklearn
.metrics.roc_auc_score.html (Links to an external site.)
"""

credit_df = pd.read_excel("default of credit card clients.xls")

credit_df.replace(["NaN", "NaT"], np.nan, inplace=True)
credit_df = credit_df.dropna(how="any", axis=0)

credit_df.rename(columns={"default payment next month": "default_payment"},
                 inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    credit_df.drop(["default_payment"], axis=1),
    credit_df["default_payment"], test_size=.30,
    random_state=0)

"""
2. Random Forest Classifier - Base Model:

Start by creating a simple Random Forest only using default parameters.

a) Use the RandomForestClassifier in sklearn. Fit your model on the training 
data.

b) Use the fitted model to predict on test data. Use the .predict_proba() and 
the .predict() methods to get predicted probabilities as well as predicted 
classes.

c) Calculate the confusion matrix and classification report (both are in 
sklearn.metrics). These are the same tools from HW #3.

d) Calculate the roc_auc_score for this model. There are many ways to do this, 
but an example is to use the probabilities from step B and utilize the 
roc_auc_score from sklearn. 
"""

rf_mod_base = RandomForestClassifier().fit(X_train, y_train)
y_test_prob_base = rf_mod_base.predict_proba(X_test)
y_test_pred = rf_mod_base.predict(X_test)

labels_df = credit_df.loc[:, credit_df.columns == "default_payment"]
labels = labels_df.drop_duplicates().sort_values("default_payment")

conf_mat = metrics.confusion_matrix(y_true=y_test,
                                    y_pred=y_test_pred)
plt.title("Confusion Matrix")
sns.heatmap(conf_mat, annot=True, fmt="d")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print(metrics.classification_report(y_test, y_test_pred))

# limit to probability for class = 1
rf_mod_base_probs = rf_mod_base.predict_proba(X_test)[:, 1]

# calculate roc_auc_score
print(metrics.roc_auc_score(y_test, rf_mod_base_probs))

"""
3. Random Forest Classifier - Grid Search:

Start by creating a simple Random Forest only using default parameters.

a) Use the RandomForestClassifier along with the GridSearchCV tool. Run the 
GridSearchCV using the following: 

n_estimators: 500, 750, 1000
max_features: 2, 4, 6 
Note: Feel free to try out more parameters, the above is the bare minimum for 
this assignment.

Use 5 cross-fold and for scoring use "roc_auc" (this is the score that will be 
referenced when identifying the best parameters). 

Example of GridSearchCV: 
#create a dictionary of parameters 
param_grid = {'max_depth':[2, 4, 6, 8],
'min_samples_split':[3,4,5,6,7,8],
'random_state':[0]}

# create Random Forest model 
rf_obj=RandomForestClassifier()

# Create gridsearch object with various combinations of parameters
rf_Grid = GridSearchCV(rf_obj, param_grid, cv = 5, 
scoring = 'roc_auc',refit = True, n_jobs=-1, verbose = 5)

# next, just fit this object

b) Identify the best performing model:

.best_params_() : This method outputs to best performing parameters

.best_estimator_() : This method outputs the best performing model, 
and can be used for predicting on the X_test 

c) Use the best estimator model to predict on test data. Use 
the .predict_proba() and the .predict() methods to get predicted 
probabilities as well as predicted classes.

d) Calculate the confusion matrix and classification report (both are in 
sklearn.metrics). 

e) Calculate the roc_auc_score for this model. 

"""

# create a dictionary of parameters
param_grid = {"n_estimators": [500, 750, 1000], "max_features": [2, 4, 6],
              "random_state": [0]}

# create random forest model
rf_obj = RandomForestClassifier()

# Create gridsearch object with various combinations of parameters
rf_grid = GridSearchCV(rf_obj, param_grid, cv=5, scoring="roc_auc",
                       refit=True, n_jobs=-1, verbose=5)

rf_grid_fit = rf_grid.fit(X_train, y_train)

rf_grid_fit.best_params_
rf_grid_fit.best_estimator_

y_test_pred_grid = rf_grid_fit.predict(X_test)
y_test_probs = rf_grid_fit.predict_proba(X_test)


conf_mat = metrics.confusion_matrix(y_true=y_test,
                                    y_pred=y_test_pred)
plt.title("Confusion Matrix")
sns.heatmap(conf_mat, annot=True, fmt="d")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print(metrics.classification_report(y_test, y_test_pred_grid))

# Limit to probability for class = 1
rf_mod_probs = rf_grid_fit.best_estimator_.predict_proba(X_test)[:, 1]

# Calculate roc_auc_score
print(metrics.roc_auc_score(y_test, y_test_probs[:, 1]))

# Feature importance
rf_grid_fit.i


