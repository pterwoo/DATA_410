# Project 4

# Boosting Algorithms 

Boosting is a method of reducing bias and variance to increase overall performance accuracy. It usually involves iteratively learning weak classifiers then adding it to one final strong classifier. 

This project will examine the performances of various regressors, including a comparison between locallyweighted regression and boosted locally weighted regression. In addition, we will also compare the performances of XGBoost, a regressor that has been consistently the most accurate in past projects, and LightGBM, another algorithm that makes use of gradient boosting known to be highly accurate. 

# Implementing Boosting

These are the necessary packages to execute the code in the project. 

```
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.linear_model import Ridge
import xgboost as xgb
```

First, we will introduce some kernels. 

```
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
```

We will now define our locally weighted regresor we have made repeated use of in past projects

```
#Defining the kernel local regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```

To compare the performance of regular locally weighted regression and boosted locally weighted regression, we introduce the booster function that performs 
the boosting, and the boosted locally weigthed regressor. 

Boosted LOESS: 

```
def boosted_lwr(X, y, xnew, kern, tau, intercept, model_boosting, nboost):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # Now train the Decision Tree on y_i - F(x_i)
  #new_y = y - Fx
  #model = DecisionTreeRegressor(max_depth=2, random_state=123)
  #model = RandomForestRegressor(n_estimators=100,max_depth=2)
  #model = model_xgb
  #model_boosting.fit(X,new_y)
  output = booster(X,y,xnew,kern,tau,model_boosting,nboost)
  return output 
```

The booster: 

```
def booster(X, y, xnew, kern, tau, intercept, model_boosting, nboost):
  Fx = lw_reg(X,y,X,kern,tau,True)
  Fx_new = lw_reg(X,y,xnew,kern,tau,True)
  new_y = y - Fx
  output = Fx
  output_new = Fx_new
  for i in range(nboost):
    model_boosting.fit(X,new_y)
    output += model_boosting.predict(X)
    output_new += model_boosting.predict(xnew)
    new_y = y - output
  return output_new 
```

We will use the concrete dataset that we have explored thoroughly in class. We will set y to be the strength, and 
the x values to be cement, water, and age. 

```
data = pd.read_csv("/tmp/concrete.csv")

X = data[["cement", "water", "age"]].values
y = data["strength"].values
```

Now that we have assigned the X and y values, we will test our custom boosted LOESS alogrithm's accuracy. We will be using 
the Epanechnikov kernel with tau = 1, and the algorithm will be boosted 3 times.

```
model_boosting = RandomForestRegressor(n_estimators=100,max_depth=3)

kf = KFold(n_splits=10, shuffle=True, random_state = 100)
scale = StandardScaler()

mse_rid = [] 
mse_brid = []
mse_lwr = []
mse_blwr = []
mse_xgb = []

# this is the Cross-Validation Loop
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  # produce y-hat values

  ##  LOESS
  yhat_lwr = lw_reg(xtrain, ytrain, xtest, Tricubic, tau = 1,intercept=True)

  ## custom boosted LOESS
  yhat_blwr =  boosted_lwr(xtrain,ytrain,xtest,Epanechnikov,1,True,model_boosting,3)

  ## XGB 
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
  model_xgb.fit(xtrain,ytrain)
  yhat_xgb = model_xgb.predict(xtest)

  # calculate MSEs
  mse_lwr.append(mse(ytest, yhat_lwr))
  mse_blwr.append(mse(ytest,yhat_blwr))
  mse_xgb.append(mse(ytest,yhat_xgb))

# print out and compre the results
print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for BLWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
```

The output: 

The Cross-validated Mean Squared Error for LWR is : 84.29482958879224
The Cross-validated Mean Squared Error for BLWR is : 57.77250519241295
The Cross-validated Mean Squared Error for XGB is : 76.33055151127176

Interestingly enough, our boosted LOESS regressor outperformed both regular LOESS and XGBoost. I have chosen the parameters 
for `boosted_lwr` randomly, which means that there is ample room for potential improvements to the model.

# LightGBM 

LightGBM is a gradient boosting framework based on decision tree algorithms used for classification, ranking, regression, 
and more. LightGBM makes use of a highly optimized histogram-based decision tree learning algorithm that compared to 
XGBoost is highly efficient. To test this, we will compare the MSE of LightGBM to that of XGBoost from the previous 
section.

To implement LightBGM, 

General imports and standard scaler
```
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error

scale = StandardScaler()
```

Cross-validate and scale dataset

```
x_train, x_test, y_train, y_test = tts(X,y)

lgb_train = lgb.Dataset(x_train, y_train)
lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train)
```

Define parameters for the regressor 

```
gbm = lgb.LGBMRegressor(boosting_type = 'gbdt',n_estimators=100, max_depth= 1)

params = {
        "objective": "regression",
        "metric": "l1",
        "num_leaves": 30,
        "learning_rate": 0.1,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": 0
    }
```

Train, predict, then calculate the mean squared error

```
gbm = lgb.train(params, train_set = lgb_train,
                valid_sets = lgb_test,
                early_stopping_rounds=1000)
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
mse_gbm =  round(mse(y_test, y_pred), 5)

print('The Cross-validated Mean Squared Error for LightGBM is : '+str((mse_gbm)))
```

The resulting MSE was 46.82874, which was lowest out of all the regressors that we have used in this project. 



