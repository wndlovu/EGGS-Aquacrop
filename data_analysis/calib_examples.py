#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:12:52 2023

@author: wayne
"""

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load the Diabetes dataset
col_name = ['age', 'sex', 'bmi' ,'map', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu'] # Declare the columns names
diabetes = datasets.load_diabetes() # Call the diabetes dataset from sklearn
#diabetes.columns = col_name

df = pd.DataFrame(diabetes.data) # load the dataset as a pandas data frame
df.columns = col_name
y = diabetes.target # define the target variable (dependent variable) as y


# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


## The line / model
plt.scatter(y_test, predictions)
plt.xlabel('Real Values')
plt.ylabel('Predictions')

print('Score:', model.score(X_test, y_test))



# k folds
from sklearn.model_selection import KFold # import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array
y = np.array([1, 2, 3, 4]) # Create another array
kf = KFold(n_splits=2) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf) 


for train_index, test_index in kf.split(X):
 print('TRAIN:', train_index, 'TEST:', test_index)
 X_train, X_test = X[train_index], X[test_index]
 y_train, y_test = y[train_index], y[test_index]

#Leave One Out Cross Validation (LOOCV)
from sklearn.model_selection import LeaveOneOut 
X = np.array([[1, 2], [3, 4]])
y = np.array([1, 2])
loo = LeaveOneOut()
loo.get_n_splits(X)


for train_index, test_index in loo.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print(X_train, X_test, y_train, y_test)


# Necessary imports: 
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

# Load the Diabetes dataset
col_name = ['age', 'sex', 'bmi' ,'map', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu'] # Declare the columns names
diabetes = datasets.load_diabetes() # Call the diabetes dataset from sklearn
#diabetes.columns = col_name

df = pd.DataFrame(diabetes.data) # load the dataset as a pandas data frame
df.columns = col_name
y = diabetes.target # define the target variable (dependent variable) as y

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

scores = cross_val_score(model, df, y, cv=6)
print('Cross-validated scores:', scores)

# Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=6)
plt.scatter(y, predictions)

#It is six times as many points as the original plot because I used cv=6
accuracy = metrics.r2_score(y, predictions)
print('Cross-Predicted Accuracy:', accuracy)


# example 2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

# training and test calibration
# there is still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally. 
X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)



# cross validation
#splitting the data, fitting a model and computing the score 5 consecutive times (with different splits each time):

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
scores

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100)

model = RandomForestRegressor()
param_search = {'n_estimators' : [10, 100]}

cv = TimeSeriesSplit(n_splits=5)
gsearch = GridSearchCV(estimator=model, cv=cv, param_grid=param_search)
gsearch.fit(X, y)

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit()
print(tscv)
TimeSeriesSplit(max_train_size=None, n_splits=3)
for train_index, test_index in tscv.split(X):
    print('TRAIN:', train_index, 'TEST:', test_index) 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
tscv = TimeSeriesSplit(n_splits = 4)
rmse = []
for train_index, test_index in tscv.split(cross_validation):
    cv_train, cv_test = cross_validation.iloc[train_index], cross_validation.iloc[test_index]
    
    arma = statsmodels.tsa.arima.model.ARIMA(cv_train, (2,2)).fit(disp=False)
    
    predictions = arma.predict(cv_test.index.values[0], cv_test.index.values[-1])
    true_values = cv_test.values
    rmse.append(sqrt(mean_squared_error(true_values, predictions)))
    
print("RMSE: {}".format(np.mean(rmse)))


for train_index, test_index in tscv.split(df):
    cv_train, cv_test = df.iloc[train_index], df.iloc[test_index]
    
    arma = statsmodels.tsa.arima.model.ARIMA(cv_train, (2,2)).fit(disp=False)
    
    predictions = arma.predict(cv_test.index.values[0], cv_test.index.values[-1])
    true_values = cv_test.values
    rmse.append(sqrt(mean_squared_error(true_values, predictions)))
    
print("RMSE: {}".format(np.mean(rmse)))


import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.arima_model import ARMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def _getStartingVals(self):
    if self._data is not None:
        if self._include_constant:
            c = 'c'
        else:
            c = 'nc'
        try:
            # statsmodels.tsa.arima_model.ARMA has been discontinued and replaced with statsmodels.tsa.arima.model.ARIMA, just force d=0 to make it an ARMA
            arma = ARIMA(self._data.values, order=(self._order['AR'],0,self._order['MA']), trend=c)
            model = arma.fit()
            # the original and now-deprecated statsmodels.tsa.arima_model.ARMA function does NOT include sigma2 in its params when returning the fitted results, so filter it out
            self._startingValues = [param for param,name in zip(model.params,arma.param_names) if name!='sigma2']
        except ValueError:
            self._startingValues = None            
    else:
        self._startingValues = np.zeros((self._pnum,))+0.5


df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
df.head()


tscv = TimeSeriesSplit(n_splits = 4)
rmse = []
for train_index, test_index in tscv.split(df):
    cv_train, cv_test = df.iloc[train_index], df.iloc[test_index]
    model = statsmodels.tsa.arima.model.ARIMA(cv_train.value, order=(0, 1)).fit()
    predictions = model.predict(cv_test.index.values[0], cv_test.index.values[-1])
    true_values = cv_test.value
    rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))



df = pd.read_csv('/Users/wayne/Downloads/Gemini_ETHUSD_d.csv', skiprows=1)
for i in range(1, len(df)):
    col_name = 'd{}'.format(i)
    df[col_name] = df['d0'].shift(periods=-1 * i)
df = df.dropna()

model = build_model(_alpha=1.0, _l1_ratio=0.3)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring=r2)
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))







