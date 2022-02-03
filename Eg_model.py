# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:56:18 2020
Modified on Thu Feb 3 10:06:00 2022

@author: Ya Zhuo, University of Houston
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
import pickle
from sklearn.svm import SVR
import pymatgen as mg
from pymatgen.core.composition import Composition

#read data
DE_c = pd.read_excel('Training_Set.xlsx',sheet_name=0)
array_c = DE_c.values
X_c = array_c[:,3:139]
Y_c = array_c[:,2]
Y_c = Y_c.astype('int')
#train/test split
X_train_c, X_test_c, Y_train_c, Y_test_c = train_test_split(X_c, Y_c, test_size=0.1, random_state=15, shuffle=True)
#preprocessing
scaler_c = preprocessing.StandardScaler().fit(X_train_c)
X_train_c = scaler_c.transform(X_train_c)
X_test_c = scaler_c.transform(X_test_c)
#model construction
classification=SVC(kernel='rbf',C=10**1.5, gamma= 0.01).fit(X_train_c, Y_train_c)
#save model
classification_model=pickle.dumps(classification)

#read data
DE_r = pd.read_excel('Training_Set.xlsx',sheet_name=1)
array_r = DE_r.values
X_r = array_r[:,2:138]
Y_r = array_r[:,1]
#train/test split
X_train_r, X_test_r, Y_train_r, Y_test_r = train_test_split(X_r, Y_r, test_size=0.1, random_state=15, shuffle=True)
#preprocessing
scaler_r = preprocessing.StandardScaler().fit(X_train_r)
X_train_r = scaler_r.transform(X_train_r)
X_test_r = scaler_r.transform(X_test_r)
#model construction
regression = SVR(kernel='rbf',C=10, epsilon=0.1, gamma= 0.01).fit(X_train_r, Y_train_r)
predicted_Y1_r = regression.predict(X_train_r)
predicted_Y_r = regression.predict(X_test_r)
#save model
regression_model=pickle.dumps(regression)

prediction = pd.read_excel('to_predict.xlsx')
prediction.head()
prediction.dtypes

class Vectorize_Formula:
	def __init__(self):
		elem_dict = pd.read_excel(r'elements.xlsx')
		self.element_df = pd.DataFrame(elem_dict) 
		self.element_df.set_index('Symbol',inplace=True)
		self.column_names = []
		for string in ['avg','diff','max','min']:
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
					print('The element:', key, 'from formula', formula,'is not currently supported in our database')
					return np.array([np.nan]*len(self.element_df.iloc[0])*4)
			max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
			min_feature = self.element_df.loc[list(fractional_composition.keys())].min()

			features = pd.DataFrame(np.concatenate([avg_feature, diff_feature, np.array(max_feature), np.array(min_feature)]))
			features = np.concatenate([avg_feature, diff_feature, np.array(max_feature), np.array(min_feature)])
			return features.transpose()
		except:
			print('There was an error with the Formula: '+ formula + ', this is a general exception with an unkown error')
			return [np.nan]*len(self.element_df.iloc[0])*4

gf=Vectorize_Formula()
features=[]
targets=[]
for formula in prediction['Composition']:
    features.append(gf.get_features(formula))
X = pd.DataFrame(features, columns = gf.column_names)
pd.set_option('display.max_columns', None)

X_c = scaler_c.transform(X)
pred_c = pickle.loads(classification_model)
result_c = pred_c.predict(X_c)
X_r = scaler_r.transform(X)
pred_r = pickle.loads(regression_model)
result_r = pred_r.predict(X_r)
result=[]
for i in range(len(result_c)):
    if result_c[i] == 1:
        result.append(result_r[i]);
    else:
        result.append(result_c[i])
result = np.around(result,decimals=2)

composition=pd.read_excel('to_predict.xlsx',sheet_name='Sheet1', usecols="A")
composition=pd.DataFrame(composition)
result=pd.DataFrame(result)
predicted=np.column_stack((composition,result))
predicted=pd.DataFrame(predicted)
predicted.to_excel('predicted.xlsx', index=False, header=("Composition","Predicted Eg"))
print("A file named predicted.xlsx has been generated.\nPlease check your folder.")
