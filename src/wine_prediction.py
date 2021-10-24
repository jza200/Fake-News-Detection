import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#Load wine data from an external URL
data = pd.read_csv('D:\\Data\\winequality-white.csv')
#data = pd.read_csv('D:\\Data\\winequality-red.csv')

#Output 5 rows of data
#print(data.head())


#Read the csv file with ; as the seperator
data = pd.read_csv('D:\\Data\\winequality-white.csv', sep=';')
print(data.head())

#quality is our target
#print(data.shape)
#print(data.describe())

#Split data into training and test sets. 20% test 80% train
y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

#Standarize the data with transformer API 
scaler = preprocessing.StandardScaler().fit(X_train) #Fitting the API
X_train_scaled = scaler.transform(X_train) #applying the transformer to training data
X_test_scaled = scaler.transform(X_test)

#pipeline the process
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
#print(pipeline.get_params())

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}


#perform k-fold cross validation for the classifier
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
#Fit and tune model
clf.fit(X_train, y_train)    
print(clf.best_params_)

#print(clf.refit)


#Predicting new set of data
y_pred = clf.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))