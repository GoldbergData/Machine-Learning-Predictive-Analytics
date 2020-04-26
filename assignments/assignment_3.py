import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing

# 1. Data Processing:
# a) Import the data: Only keep numeric data (pandas has tools to do this!).
# Drop "PHONE" and "COUNTRY_SSA" as well.
# b) This data is extra messy and has some NaN and NaT values. NaT values
# should be replaced by "np.nan." After this step, remove any rows that have
# an NaN value.
# c) Split into train / test set using an 80/20 split.
# d) Scale all input features (NOT THE TARGET VARIABLE)

provider_df = pd.read_csv("ProviderInfo.csv")

provider_df_num = provider_df.select_dtypes(["number"])

provider_df_num.drop(["PHONE", "COUNTY_SSA"], axis=1, inplace=True)

provider_df_num.replace(["NaN", "NaT"], np.nan, inplace=True)
provider_clean_df = provider_df_num.dropna(how="any", axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    provider_clean_df.drop(["OVERALL_RATING"], axis=1),
    provider_clean_df["OVERALL_RATING"], test_size=.20,
    random_state=0)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 2. Model #1: Logistic Regression
# a) Pick up from step d in Problem 1 (use the same data that has been
# scaled): Using LogisticRegression(), build a model to predict the
# "OVERALL_RATING". Note: The default in sklearn is "one-vs-rest"
# classification, where we calculate the probability of each class compared
# to the rest. This is fine for the homework!
# Read more in documentation:
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model
# .LogisticRegression.html
# b) For error evaluation, start by calculating the score (returns the mean
# accuracy).
# Example: Off of your logistic regression model, run the .score() method
# c) Calculate the confusion matrix and classification report (both are in
# sklearn.metrics).

logit_mod = LogisticRegression(multi_class="ovr").fit(X_train, y_train)

y_train_pred = logit_mod.predict(X_train)
y_test_pred = logit_mod.predict(X_test)

labels_df = provider_clean_df.loc[:,
            provider_clean_df.columns == "OVERALL_RATING"]
labels = labels_df.drop_duplicates(
).sort_values("OVERALL_RATING")
labels = labels.assign(label=["One", "Two", "Three", "Four", "Five"])

conf_mat = metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt="d",
            xticklabels=labels.label.values,
            yticklabels=labels.label.values)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print(metrics.classification_report(y_test, y_test_pred,
                                    target_names=labels.label.values))

# 3. Model #2: PCA(n_components = 2) + Logistic Regression
#
# a) Pick up from step d in Problem 1 (use the same data that has been
# scaled): We will now transform the X_train & X_test data using PCA with 2
# components.
#
# Example:
#
# # import PCA object from sklearn
# from sklearn.decomposition import PCA
#
# # limit PCA object to 2 components
# pca_two = PCA(n_components=2)
#
# use pca object to fit & apply pca transformation to data X_train_pca =
# pca_two.fit_transform(X_train_scaled)
# b) Then use the transformed data (X_train_pca) to fit a Logistic Regression
#  model.
#
# c) Calculate the same error metrics as those from Model #1.

# limit PCA object to 2 components
pca_two = PCA(n_components=2).fit(X_train)

# use pca object to fit & apply pca transformation to data
X_train_pca2 = pca_two.transform(X_train)
X_test_pca2 = pca_two.transform(X_test)

logit_mod_pca = LogisticRegression(multi_class="ovr").fit(X_train_pca2, y_train)
y_test_pred_pca2 = logit_mod_pca.predict(X_test_pca2)

conf_mat = metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred_pca2)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt="d",
            xticklabels=labels.label.values,
            yticklabels=labels.label.values)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print(metrics.classification_report(y_test, y_test_pred_pca2,
                                    target_names=labels.label.values))

# 4. Model #3: PCA(n_components = 16) + Logistic Regression
# a) Pick up from step d in Problem 1 (use the same data that has been
# scaled): We will now transform the X_train & X_test data using PCA with 16
# components.
# b) Then use the transformed data (X_train_pca) to fit a Logistic Regression
# model.
# c) Calculate the same error metrics as those from Model #1.

pca_sixteen = PCA(n_components=16).fit(X_train)

# use pca object to fit & apply pca transformation to data
X_train_pca16 = pca_sixteen.transform(X_train)
X_test_pca16 = pca_sixteen.transform(X_test)

logit_mod_pca = LogisticRegression(multi_class="ovr").fit(X_train_pca16,
                                                          y_train)
y_test_pred_pca16 = logit_mod_pca.predict(X_test_pca16)

conf_mat = metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred_pca16)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt="d",
            xticklabels=labels.label.values,
            yticklabels=labels.label.values)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print(metrics.classification_report(y_test, y_test_pred_pca16,
                                    target_names=labels.label.values))
