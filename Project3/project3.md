# Project 3

## Extreme Gradient Boosting

Extreme Gradient Boosting, also referred to as XGBoost, is a form of gradient boosting algorithm commonly used in classification and regression problems. 
Gradient boosting makes use of decision tree models added to ensembles, correcting prediction errors made by prior models. These models are fit using a gradient descent
optimization algorithm. Extreme gradient boosting differ from conventional gradient boosting algorithms in that they incorporate regularization parameters
that help in preventing overfitting. 

### Applying XGBoost

To begin, we must import XGBoost and select our data. We will be using the cars dataset that we have used previously. The X values will be engine (ENG), number of cylinders (CYL), and weight (WGT). The y value will be miles per gallon (MPG)

```
import xgboost as xgb

cars = pd.read_csv("/tmp/cars.csv")

X = cars[['ENG','CYL','WGT']].values
y = cars['MPG'].values
```

We can create a loop to repeat the process multiple times to get an average MSE for better comparison. 

```
mse_xgb = []

for i in range(5):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    mse_xgb.append(mse(ytest,yhat_xgb))

print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
```
The output from the code above:
`The Cross-validated Mean Squared Error for XGB is : 16.559417572167884`

## Locally Weighted Regression (LOESS)

Locally Weighted Regression, also referred to as LOESS or LOWESS, is a regression method that fits a local set of points using weighted least squares.The weights are determined by kernel functions. The kernel function attributes more weight to points near the tarrget point and less weight to points that are further away. It has discernible advantages compared to other forms of regression. The model if very flexible, allowing complex processes to be modeled and fitted.

### Applying LOESS

We need to define a kernel to be used in the regressor. 

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

We can define the actual model:

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

In fitting and cross-validating the model, we will be using the same data and the same code format from above for easier comparison. 

```
mse_lwr = []

for i in range(5):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
    yhat_xgb = model_xgb.predict(xtest)
    mse_lwr.append(mse(ytest,yhat_lwr))

print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
```
The output:
`The Cross-validated Mean Squared Error for LWR is : 16.927710396099975`

## Boosted Locally Weighted Regression

We will also try a boosted version of the locally weighted regressor to see how it compares to XGBoost and non-boosted locally weighted regression.

### Applying Boosted LOESS

We will first define a function that boosts the model:

```
def booster(X,y,xnew,kern,tau,model_boosting,nboost):
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

Then, using the function defined above, we can create a boosted version of the LOESS model. 

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

Finally, we can use the same code format to calculate the average MSE.

```
mse_blwr = []

for i in range(5):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    yhat_blwr = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    mse_lwr.append(mse(ytest,yhat_lwr))
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_xgb.append(mse(ytest,yhat_xgb))

print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Boosted LWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
```

Output:
`The Cross-validated Mean Squared Error for Boosted LWR is : 16.662661181680825`

## Results 

Comparing the MSEs of the three models, we can observe that the XGBoost model worked best, then boosted LOESS, then normal LOESS. 
These results are to be expected, since typically boosting a model will increase the accuracy of the model. It is interesting to note that XGBoost worked better than boosted LOESS. We have discussed in class that in regression problems, XGBoost will likely have the best performance. While this experiment only recorded an average of 5 iterations of the model, we have observed that XGBoost had an advantage compared against the other models. 

## Works Cited 

Brownlee, J. (2021, April 26). Extreme gradient boosting (XGBoost) ensemble in Python. Machine Learning Mastery. Retrieved February 28, 2022, from https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/#:~:text=Extreme%20Gradient%20Boosting%20is%20an,with%20the%20scikit%2Dlearn%20API. 

Sicotte, X. B. (2018, May 24). Xavier Bourret Sicotte. Locally Weighted Linear Regression (Loess) - Data Blog. Retrieved February 13, 2022, from https://xavierbourretsicotte.github.io/loess.html 
