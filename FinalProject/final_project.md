# Final Project - Predicting Term Deposit Subscription with Marketing Data
#### Jordan Landrum, Minkyong Song, Peter Woo

## Introduction

For this analysis, the ‘Bank Marketing Data Set’ was acquired from the UCI Machine Learning Repository. 
This data set is based off of a Portugues banking institution that contacted potential customers with the goal of having them arrange to receive a term deposit.
A term deposit is when you deposit a sum of money into the bank for a period of time to ‘mature’ so it can gain interest (Kagan 2021). When making a term deposit, 
it is different from opening a checking or savings account, since the money is not available for withdrawal until after the term date has passed (Kagan 2021). 
Machine Learning methods of classification and prediction will be cross-validated using logistic regression, light gbm, extreme gradient boosting, and neural networks 
in search of the most accurate model of the data.

## Data and Pre-Processing

In the full bank marketing data set, there are 20 input variables (x’s) and 1 target variable (y). The input variables consist of: 
* age: The age of the client.
* job: The type of job.
* marital: The marital status of the client.
* education: The level of education the client has received.
* default: Whether the client has credit in default.
* housing: Whether the client has a housing loan.
* loan: Whether the client has a personal loan.
* contact: The method of communication.
* month: The month last contacted.
* day_of_week: The day of the week last contacted.
* duration: The duration of the last contact in seconds.
* campaign: The number of contacts performed during the campaign.
* pdays: The number of days that passed by after the client was last contacted.
* previous: The number of contacts performed before the campaign for the client.
* poutcome: Outcome of the previous marketing campaign.
* emp.var.rate: Employment variation rate of Portugal
* cons.price.ind: Consumer price index of Portugal
* cons.conf.ind: Consumer confidence index of Portugal
* euribor3m: Euribor month rate of Portugal
* nr.employed: The number of employees in Portugal

Whether the client subscribed a term deposit (y/n) is the target variable (y) for this dataset. 

To process this data for classification, we begin by importing the necessary packages 

```
# import all necessary packages 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression #, LinearRegression, Lasso, Ridge, ElasticNet
import xgboost as xgb


from sklearn.metrics import accuracy_score as AC, confusion_matrix as CM
from matplotlib.colors import ListedColormap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.datasets import make_classification
```

We read in the data:

```
data_full = pd.read_csv('drive/My Drive/Colab Notebooks/FINAL_PROJECT/bank-additional/bank-additional-full.csv', delimiter = ';')
```

The main objective is to transform the categorical variables into either 1 or 0 as a dummy variable for each of its category.

These are the name of the categorical variables in the data set:
- job
- marital
- education
- default
- housing
- loan
- contact
- month
- day_of_week
- poutcome
- y

To do this: 

```
# create dataframes with dummy variables 
job = pd.get_dummies(data_full['job'], prefix = 'job')
marital = pd.get_dummies(data_full['marital'], prefix = 'marital')
education = pd.get_dummies(data_full['education'], prefix = 'edu')
default = pd.get_dummies(data_full['default'], prefix = 'default')
housing = pd.get_dummies(data_full['housing'], prefix = 'housing')
loan = pd.get_dummies(data_full['loan'], prefix = 'loan')
contact = pd.get_dummies(data_full['contact'], prefix = 'contact')
month = pd.get_dummies(data_full['month'], prefix = 'month')
day = pd.get_dummies(data_full['day_of_week'], prefix = 'day')
poutcome = pd.get_dummies(data_full['poutcome'], prefix = 'poutcome')

# create a separate dataframe just for the target variable
target = pd.get_dummies(data_full['y'], drop_first = True)
target.columns = ['target'] # rename column name to 'target'

# drop the original columns, concatenate the dummy columns
data_full.drop(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y'], axis = 1, inplace = True)
data_processed = pd.concat([data_full, job, marital, education, default, housing, loan, contact, month, day, poutcome, target], axis = 1)
```

With this, the preprocessing is complete. 

### Feature Selection 

Since the resulting dataframe from adding all dummy variables contains 64 features, selecting only important features felt important. 
Through research, two feature selection methods were implemented - Logistic classification feature importance, and XGBoost classification feature importance. 

#### Logistic classification feature importance 

This method can be used to determine feature importance by fitting the Logistic Regression model and retrieving the `coeff_` property. This property 
represents a feature importance score.

To begin, the X and y varaibles must be defined. 

```
X = data_processed.drop('target', 1)
y = data_processed[['target']]
```

Now X is set as every feature variable, and y is set as the "target" variable defined previously. 

```
model = LogisticRegression()

# fit the model
model.fit(X, y)

# get importance
importance_logistic = model.coef_[0]

# plot feature importance
pyplot.bar([x for x in range(len(importance_logistic))], importance_logistic)
pyplot.show()
```

The resulting plot: 

![](FinalProject/importance_log.jpg)

In addition to the plot, the summary of the feature scores are obtained through the following: 

```
for i,v in enumerate(importance_logistic):
	print('Feature: %0d, Score: %.5f' % (i,v))
```

The summary: 

Feature: 0, Score: 0.00286
Feature: 1, Score: 0.00454
Feature: 2, Score: 0.04899
Feature: 3, Score: -0.00151
Feature: 4, Score: -0.08000
Feature: 5, Score: -0.19280
Feature: 6, Score: 0.40510
Feature: 7, Score: 0.03939
Feature: 8, Score: -0.22470
Feature: 9, Score: -0.00746
Feature: 10, Score: 0.05998
Feature: 11, Score: -0.11652
Feature: 12, Score: -0.01035
Feature: 13, Score: -0.00171
Feature: 14, Score: 0.00170
Feature: 15, Score: 0.04273
Feature: 16, Score: -0.00136
Feature: 17, Score: -0.03151
Feature: 18, Score: 0.03261
Feature: 19, Score: 0.02150
Feature: 20, Score: 0.00618
Feature: 21, Score: 0.00027
Feature: 22, Score: -0.00398
Feature: 23, Score: -0.06719
Feature: 24, Score: 0.07476
Feature: 25, Score: -0.00007
Feature: 26, Score: -0.02419
Feature: 27, Score: -0.01935
Feature: 28, Score: -0.06007
Feature: 29, Score: -0.02476
Feature: 30, Score: 0.00087
Feature: 31, Score: 0.01430
Feature: 32, Score: 0.10537
Feature: 33, Score: 0.01134
Feature: 34, Score: 0.09665
Feature: 35, Score: -0.09310
Feature: 36, Score: -0.00003
Feature: 37, Score: -0.00190
Feature: 38, Score: -0.00249
Feature: 39, Score: 0.00790
Feature: 40, Score: 0.01653
Feature: 41, Score: -0.00249
Feature: 42, Score: -0.01053
Feature: 43, Score: 0.11760
Feature: 44, Score: -0.11409
Feature: 45, Score: 0.02374
Feature: 46, Score: 0.04728
Feature: 47, Score: 0.00471
Feature: 48, Score: 0.07925
Feature: 49, Score: 0.06232
Feature: 50, Score: 0.08734
Feature: 51, Score: -0.30821
Feature: 52, Score: -0.01967
Feature: 53, Score: 0.02177
Feature: 54, Score: 0.00499
Feature: 55, Score: -0.01688
Feature: 56, Score: -0.03416
Feature: 57, Score: 0.00691
Feature: 58, Score: 0.02389
Feature: 59, Score: 0.02375
Feature: 60, Score: -0.09471
Feature: 61, Score: 0.08192
Feature: 62, Score: 0.01631

#### XGBost classification feature importance

Much like Logistic Regression, it is simple to pull the feature importance score once the XGBoost model is fit. The scores are saved in the `feature_importances_`
property. 

```
model = XGBClassifier()

# fit the model
model.fit(X, y)

# get importance
importance_xgb = model.feature_importances_

# plot feature importance
pyplot.bar([x for x in range(len(importance_xgb))], importance_xgb)
pyplot.show()
```

The resulting plot: 

![](FinalProject/importance_xgb.jpg)

A summary can be outputted using the same method as above: 

```
for i,v in enumerate(importance_xgb):
	print('Feature: %0d, Score: %.5f' % (i,v))
```

Feature: 0, Score: 0.00835
Feature: 1, Score: 0.12761
Feature: 2, Score: 0.00628
Feature: 3, Score: 0.03964
Feature: 4, Score: 0.00782
Feature: 5, Score: 0.04754
Feature: 6, Score: 0.03001
Feature: 7, Score: 0.05282
Feature: 8, Score: 0.03656
Feature: 9, Score: 0.38818
Feature: 10, Score: 0.00194
Feature: 11, Score: 0.01221
Feature: 12, Score: 0.00000
Feature: 13, Score: 0.00000
Feature: 14, Score: 0.00000
Feature: 15, Score: 0.00000
Feature: 16, Score: 0.00000
Feature: 17, Score: 0.00000
Feature: 18, Score: 0.00109
Feature: 19, Score: 0.00341
Feature: 20, Score: 0.00000
Feature: 21, Score: 0.00000
Feature: 22, Score: 0.00000
Feature: 23, Score: 0.00000
Feature: 24, Score: 0.00640
Feature: 25, Score: 0.00000
Feature: 26, Score: 0.00508
Feature: 27, Score: 0.00000
Feature: 28, Score: 0.00000
Feature: 29, Score: 0.00473
Feature: 30, Score: 0.00000
Feature: 31, Score: 0.00000
Feature: 32, Score: 0.00903
Feature: 33, Score: 0.00000
Feature: 34, Score: 0.01620
Feature: 35, Score: 0.00000
Feature: 36, Score: 0.00000
Feature: 37, Score: 0.00000
Feature: 38, Score: 0.00000
Feature: 39, Score: 0.00000
Feature: 40, Score: 0.00000
Feature: 41, Score: 0.00000
Feature: 42, Score: 0.00000
Feature: 43, Score: 0.02574
Feature: 44, Score: 0.00000
Feature: 45, Score: 0.00927
Feature: 46, Score: 0.00000
Feature: 47, Score: 0.00000
Feature: 48, Score: 0.00000
Feature: 49, Score: 0.00000
Feature: 50, Score: 0.01631
Feature: 51, Score: 0.03707
Feature: 52, Score: 0.00681
Feature: 53, Score: 0.03293
Feature: 54, Score: 0.00000
Feature: 55, Score: 0.00300
Feature: 56, Score: 0.01013
Feature: 57, Score: 0.00537
Feature: 58, Score: 0.00000
Feature: 59, Score: 0.00354
Feature: 60, Score: 0.01243
Feature: 61, Score: 0.00000
Feature: 62, Score: 0.03251

Since using the XGBoost method results in a rather even split between scores that are zero and non-zero, we decided to use this method and select only features that
returned a positive score. 

```
selected_ids = []
counter = 0

#loop through the list of scores, only select non-zero ones.
for i in importance_xgb:
  if i > 0: 
    selected_ids.append(counter)
  counter += 1

# subset final dataframe
reduced = X.iloc[:,selected_ids]
```

## Classification 

### Logistic Regression

To apply Logistic Regression: 

```
model = LogisticRegression(solver='lbfgs')

cv = StratifiedKFold(n_splits=10)

log_accur = []

# create train-test-split, fit the model, then calculate its MSE. 
for idxtrain, idxtest in cv.split(X,y):
  Xtrain = X.iloc[idxtrain]
  Xtest = X.iloc[idxtest]
  ytrain = y.iloc[idxtrain]
  ytest = y.iloc[idxtest]
  model.fit(Xtrain, ytrain)
  pred = model.predict(Xtest)
  log_accur.append(AC(ytest,pred))
  
print('\n the 10-fold cross-validated Mean Squared Error of Logistic Regression is ' + str(np.mean(log_accur)) + '\n')
```

Using this method, the accuracy of Logistic Regression was 0.15199497796314854. 

### XGBoost 

```
# select the model and its parameters
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=1,alpha=1,gamma=1,max_depth=10)

mse_xgb = []

kf = KFold(n_splits=10, shuffle=True, random_state=123)

# create K-fold, calculate MSE
for idxtrain, idxtest in kf.split(X):
  Xtrain = X.iloc[idxtrain]
  Xtest = X.iloc[idxtest]
  ytrain = y.iloc[idxtrain]
  ytest = y.iloc[idxtest]
  model_xgb.fit(Xtrain, ytrain)
  yhat_xgb = model_xgb.predict(Xtest)
  mse_xgb.append(mse(ytest, yhat_xgb))

print('\n the 10-fold cross-validated Mean Squared Error of XGBoost is ' + str(np.mean(mse_xgb)) + '\n')
```
the 10-fold cross-validated Mean Squared Error of XGBoost is 0.055775859040606356, a significant improvement from Logistic Regression. 
 
### LightGBM 
 
``` 
# import the LightGBM package
import lightgbm as lgb

# select model and parameters
model_gbm = lgb.LGBMRegressor(boosting_type = 'gbdt',n_estimators=100, max_depth= 1)

mse_gbm = []

kf = KFold(n_splits=10, shuffle=True, random_state=123)

# calculate 10-fold cross validated MSE
for idxtrain, idxtest in kf.split(X):
  Xtrain = X.iloc[idxtrain]
  Xtest = X.iloc[idxtest]
  ytrain = y.iloc[idxtrain]
  ytest = y.iloc[idxtest]
  model_gbm.fit(Xtrain, ytrain)
  yhat_gbm = model_gbm.predict(Xtest)
  mse_gbm.append(mse(ytest, yhat_gbm))
 
print('\n the 10-fold cross-validated Mean Squared Error of LightGBM is ' + str(np.mean(mse_gbm)) + '\n')
```

the 10-fold cross-validated Mean Squared Error of LightGBM is 0.06362633832605548. We see that this value is very close to the MSE of XGBoost. 

#### Neural Networks 

```
from numpy import loadtxt
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LeakyReLU, ReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.preprocessing import StandardScaler
```

Building the model:

```
# create the model
model = Sequential()

model.add(Dense(100, input_dim=X.shape[1], activation='relu'))

# add hidden layers with rectified linear unit and sigmoid activation functions
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# optimizer
opt = Adam(learning_rate=0.01)

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Evaluating the model:

```
nn_accur = []

# calculate cross-validated MSE
for idxtrain, idxtest in cv.split(X,y):
  Xtrain = X.iloc[list(idxtrain)]
  Xtest = X.iloc[list(idxtest)]
  ytrain = y.iloc[list(idxtrain)]
  ytest = y.iloc[list(idxtest)]
  model.fit(Xtrain,ytrain)
  pred = model.predict(Xtest)
  accuracy = mse(ytest,model.predict(Xtest))
  nn_accur.append(accuracy)

print('\n the 10-fold cross-validated Mean Squared Error of the Neural Network model is ' + str(np.mean(nn_accur)) + '\n')
```

1159/1159 [==============================] - 2s 2ms/step - loss: 0.2097 - accuracy: 0.9020
1159/1159 [==============================] - 2s 2ms/step - loss: 0.2087 - accuracy: 0.8998
1159/1159 [==============================] - 2s 2ms/step - loss: 0.2043 - accuracy: 0.9014
1159/1159 [==============================] - 3s 2ms/step - loss: 0.2055 - accuracy: 0.9032
1159/1159 [==============================] - 3s 2ms/step - loss: 0.2109 - accuracy: 0.8971
1159/1159 [==============================] - 3s 2ms/step - loss: 0.2095 - accuracy: 0.9007
1159/1159 [==============================] - 3s 2ms/step - loss: 0.2114 - accuracy: 0.8973
1159/1159 [==============================] - 3s 2ms/step - loss: 0.2018 - accuracy: 0.9041
1159/1159 [==============================] - 3s 2ms/step - loss: 0.2063 - accuracy: 0.9023
1159/1159 [==============================] - 3s 2ms/step - loss: 0.1805 - accuracy: 0.9204

The 10-fold cross-validated Mean Squared Error of the Neural Network model is 0.0823738023884385. It performed well, but not as well as XGBoost or LightGBM.







