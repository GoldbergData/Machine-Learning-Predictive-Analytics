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
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
import warnings

# warnings.filterwarnings("ignore")

"""
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

a) Use the RandomForestClassifier in sklearn. Fit your model on the training data.

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
plt.title('Confusion Matrix')
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt="d")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print(metrics.classification_report(y_test, y_test_pred))

# limit to probability for class = 1
rf_mod_base = rf_mod_base.predict_proba(X_test)[:, 1]

# calculate roc_auc_score
# print(metrics.roc_auc_score(y_test, py_test_prob_base))
