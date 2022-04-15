# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:56:18 2020
Modified on Thu Feb 3 10:06:00 2022

@author: Ya Zhuo, University of Houston
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn import metrics
import pickle
from pymatgen.core.composition import Composition
import matplotlib.pyplot as plt

# Classification
DE_c = pd.read_excel('Training_Set.xlsx', sheet_name=0)
array_c = DE_c.values
X_c = array_c[:, 3:139]
Y_c = array_c[:, 2]
Y_c = Y_c.astype('int')

# train/test split
X_train_c, X_test_c, Y_train_c, Y_test_c = train_test_split(X_c, Y_c,  random_state=15, shuffle=True)

# preprocessing
scaler_c = preprocessing.StandardScaler()
X_train_c = scaler_c.fit_transform(X_train_c)
X_test_c = scaler_c.fit_transform(X_test_c)
# scaler_c = preprocessing.StandardScaler().fit(X_c)
X_c = scaler_c.fit_transform(X_c)

# Random forest model construction
classification_for_eval = RandomForestClassifier(n_jobs=-1, random_state=100).fit(X_train_c, Y_train_c)

# Estimation of classification model performance
y_pred_c = classification_for_eval.predict(X_test_c)
print("Random forest classification accuracy:", classification_for_eval.score(X_test_c,  Y_test_c))
print("Random forest classification precision:", metrics.precision_score(Y_test_c, y_pred_c))
print("Random forest classification recall:", metrics.recall_score(Y_test_c, y_pred_c))
print("Random forest classification F1 score:", metrics.f1_score(Y_test_c, y_pred_c))

# Grid Search to compare the performance of a random forest with different numbers of trees.
# param_grid = {
#     'n_estimators': [i for i in range(200, 301, 10)],
# }
# gs = GridSearchCV(classification, param_grid, cv=5, scoring='f1')
# gs.fit(X_train_c, Y_train_c)
# print("best params for classification:", gs.best_params_)

# Elbow Graph to define optimal number of trees in forest
# n_estimators = [i for i in range(10, 301, 10)]
# max_features = [i for i in range(1, 101, 5)]
# param_grid = {'max_features': max_features}
# gs = GridSearchCV(classification, param_grid, cv=5, scoring='f1')
# gs.fit(X_train_c, Y_train_c)
# scores = gs.cv_results_['mean_test_score']
# plt.plot(max_features, scores)
# plt.xlabel('max_features')
# plt.ylabel('f1_score')
# plt.savefig('Elbow Graph for classification_max_features.png')
# Final  model
# save model
classification = RandomForestClassifier(n_jobs=-1, random_state=100, n_estimators=100, max_features=20).fit(X_c, Y_c)
classification_model = pickle.dumps(classification)

# Regression
DE_r = pd.read_excel('Training_Set.xlsx', sheet_name=1)
array_r = DE_r.values
X_r = array_r[:, 2:138]
Y_r = array_r[:, 1]

# train/test split
X_train_r, X_test_r, Y_train_r, Y_test_r = train_test_split(X_r, Y_r, random_state=15, shuffle=True)

# preprocessing
scaler_r = preprocessing.StandardScaler()
X_train_r = scaler_r.fit_transform(X_train_r)
X_test_r = scaler_r.fit_transform(X_test_r)
# scaler_r = preprocessing.StandardScaler().fit(X_r)
X_r = scaler_r.fit_transform(X_r)

# Random forest model construction
regression_eval = RandomForestRegressor(n_jobs=-1, random_state=100).fit(X_train_r, Y_train_r)
predicted_Y1_r = regression_eval.predict(X_train_r)

# Estimation of regression model performance
y_pred_r = regression_eval.predict(X_test_r)
print("Random forest regression accuracy:", regression_eval.score(X_test_r,  Y_test_r))
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(Y_test_r, y_pred_r))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(Y_test_r, y_pred_r))
print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(Y_test_r, y_pred_r, squared=False))
print('Explained Variance Score:', metrics.explained_variance_score(Y_test_r, y_pred_r))
print('Mean Squared Log Error:', metrics.mean_squared_log_error(Y_test_r, y_pred_r))
print('Median Absolute Error:', metrics.median_absolute_error(Y_test_r, y_pred_r))

# Grid Search to compare the performance of a random forest with different numbers of trees.
# param_grid = {
#     'n_estimators': [i for i in range(50, 301, 50)],
#     'max_features': [i for i in range(1, 101, 5)]
# }
# gs = GridSearchCV(regression, param_grid, cv=5)
# gs.fit(X_train_r, Y_train_r)
# print("best params for regression:", gs.best_params_)

# Elbow Graph to define optimal number of trees in forest
# n_estimators = [i for i in range(10, 301, 50)]
# max_features = [i for i in range(1, 101, 5)]
# param_grid = {'max_features': max_features}
# gs = GridSearchCV(regression, param_grid, cv=5)
# gs.fit(X_train_r, Y_train_r)
# scores = gs.cv_results_['mean_test_score']
# # plt.plot(n_estimators, scores)
# plt.plot(max_features, scores)
# plt.xlabel('max_features')
# plt.ylabel('accuracy')
# plt.savefig('Elbow Graph for regression_max_features.png')

# Final model
regression = RandomForestRegressor(n_jobs=-1, random_state=100, n_estimators=150, max_features=25).fit(X_r, Y_r)
# save model
regression_model = pickle.dumps(regression)

prediction = pd.read_excel('to_predict.xlsx')
prediction.head()
prediction.dtypes


class VectorizeFormula:
    def __init__(self):
        elem_dict = pd.read_excel(r'elements.xlsx')
        self.element_df = pd.DataFrame(elem_dict)
        self.element_df.set_index('Symbol', inplace=True)
        self.column_names = []
        for string in ['avg', 'diff', 'max', 'min']:
            for column_name in self.element_df.columns.values:
                self.column_names.append(string+'_'+column_name)

    def get_features(self, formula):
        try:
            fractional_composition = Composition(formula).fractional_composition.as_dict()
            element_composition = Composition(formula).element_composition.as_dict()
            avg_feature = np.zeros(len(self.element_df.iloc[0]))
            sum_feature = np.zeros(len(self.element_df.iloc[0]))
            for key in fractional_composition:
                try:
                    avg_feature += self.element_df.loc[key].values * fractional_composition[key]
                    diff_feature = self.element_df.loc[list(fractional_composition.keys())].max()-self.element_df.loc[list(fractional_composition.keys())].min()
                except Exception as e:
                    print('The element:', key, 'from formula', formula, 'is not currently supported in our database')
                    return np.array([np.nan]*len(self.element_df.iloc[0])*4)
            max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
            min_feature = self.element_df.loc[list(fractional_composition.keys())].min()

            features = pd.DataFrame(np.concatenate([avg_feature, diff_feature, np.array(max_feature), np.array(min_feature)]))
            features = np.concatenate([avg_feature, diff_feature, np.array(max_feature), np.array(min_feature)])
            return features.transpose()
        except:
            print('There was an error with the Formula: ' + formula + ', this is a general exception with an unkown error')
            return [np.nan]*len(self.element_df.iloc[0])*4


gf = VectorizeFormula()
features = []
targets = []
for formula in prediction['Composition']:
    features.append(gf.get_features(formula))
X = pd.DataFrame(features, columns=gf.column_names)
pd.set_option('display.max_columns', None)

X_c = scaler_c.transform(X)
pred_c = pickle.loads(classification_model)
result_c = pred_c.predict(X_c)
X_r = scaler_r.transform(X)
pred_r = pickle.loads(regression_model)
result_r = pred_r.predict(X_r)
result = []
for i in range(len(result_c)):
    if result_c[i] == 1:
        result.append(result_r[i])
    else:
        result.append(result_c[i])
result = np.around(result, decimals=2)

composition = pd.read_excel('to_predict.xlsx', sheet_name='Sheet1', usecols="A")
composition = pd.DataFrame(composition)
result = pd.DataFrame(result)
predicted = np.column_stack((composition, result))
predicted = pd.DataFrame(predicted)
predicted.to_excel('predicted.xlsx', index=False, header=("Composition", "Predicted Eg"))
print("A file named predicted.xlsx has been generated.\nPlease check your folder.")
