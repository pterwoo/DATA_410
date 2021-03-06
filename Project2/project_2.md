# Project 2

## Locally Weighted Regression 

Locally Weighted Regression, also referred to as LOESS or LOWESS, is a regression method that fits a local set of points using weighted least squares.
The weights are determined by kernel functions. The kernel function attributes more weight to points near the tarrget point and less weight to points that are 
further away. It has discernible advantages compared to other forms of regression. The model if very flexible, allowing complex processes to be modeled and fitted.

To begin building a locally weighted regression model, we must first define some kernels. 

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
After this, we can define the model itself

```
def lowess_reg(x, y, xnew, kern, tau):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    # IMPORTANT: we expect x to the sorted increasingly
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        #theta = linalg.solve(A, b) # A*theta = b
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)
```

### Applying Locally Weighted Regression 

We will be using the mtcars dataset. The two values that we are interested in are the weights and miles per gallon. We designate the weights as the x values
and miles per gallon as y values. 

```
x = cars['WGT'].values
y = cars['MPG'].values
```

We can examine the data using a scatterplot: 

![Original Scatterplot](../images/wgt_mpg.PNG)

We will then split the values into training/testing sets, and standardize the input features using the sklearn StandardScaler. 

```
# train test split
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.25, random_state=123)

# standardizing features
scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain.reshape(-1,1))
xtest_scaled = scale.transform(xtest.reshape(-1,1))
```

Now, we can apply the model we built earlier on and examine the mean squared error. 

```
# call the function
yhat_test = lowess_reg(xtrain_scaled.ravel(),ytrain,xtest_scaled,Tricubic,0.1)

# examine the MSE
mse(yhat_test,ytest)
```

The MSE resulting from the processes above was 15.961885966790936. 

To plot this, we use the column_stack() sorting method discussed in class. 

```
# column stack then sort
M = np.column_stack([xtest,yhat_test])
M = M[np.argsort(M[:,0])]

# create the plot
plt.scatter(x,y,color='blue',alpha=0.5)
plt.plot(M[:,0],M[:,1],color='red',lw=2)
```

The resulting plot is the following: 

![Lowess Model](../images/cars_lowess.PNG)

## Random Forest Regressor 

Random forest regressor is also a commonly used form of supervised machine learning. It makes use of ensemble learning, meaning that it incorporates predictions from
a number of machine learning algorithms to create a more precise prediction. 

### Applying Random Forest Regressor 

Instead of building the model ourselves, we will use a pre-defined implementation of random forest from sklearn.  

We can continue using the scaled test and train sets from setting up LOESS. 

```
# define the model
rf = RandomForestRegressor(n_estimators=100,max_depth=3)

# fit the model
rf.fit(xtrain_scaled,ytrain)

# examine the mean squared error 
mse(ytest,rf.predict(xtest_scaled))
```

The MSE yielded with the above code is 15.998252226090576, which is slightly higher than the MSE from our LOESS model. 

To plot the model, 

```
# predict yhat
yhat = rf.predict(xtest_scaled.reshape(-1,1))

# sort
M = np.column_stack([xtest,yhat])
M = M[np.argsort(M[:,0])]

# plot 
plt.scatter(x,y,color='blue',alpha=0.5)
plt.plot(M[:,0],M[:,1],color='red',lw=2)
```

The resulting plot: 

![Random Forest](../images/cars_rf.PNG)


## Results

Judging from the mean squared error of each of the models, we are able to observe a slight advantage in favor of LOESS. However, the difference was only
marginal, therefore repeating the above processes multiple times and adjusting the parameters of the LOESS function will likely yield more conclusive results. 

## Works Cited

Bakshi, C. (2020, June 9). Random Forest regression. Medium. Retrieved February 13, 2022, from https://levelup.gitconnected.com/random-forest-regression-209c0f354c84 

Sicotte, X. B. (2018, May 24). Xavier Bourret Sicotte. Locally Weighted Linear Regression (Loess) - Data Blog. Retrieved February 13, 2022, from https://xavierbourretsicotte.github.io/loess.html 

