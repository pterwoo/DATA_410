# Project 5

## Regularization and Variable Selection Techniques 

### Square Root Lasso 

Square Root Lasso is a method that was propsed as a solution to a convex conic programming problem. It is a method for 
estimating high-dimensional sparse linear regressions that is based on minimizing an objective function plus a $L_1$ penalty. 

It is represented in the optimization problem: 

$$
\displaystyle\text{minimize} \sqrt{\frac{1}{n}\sum\limits_{i=1}^{n}(y_i-\hat{y}_i)^2} +\alpha\sum\limits_{i=1}^{p}|\beta_i|
$$

This method is a modification of the lasso, with an emphasis on dealing with problems where the number of regressors are large while only a number of them are 
significant. It is robust in that the method does not depend on prior knowledge of standard deviation or on normality. 

#### Building an sklearn-compliant implementation of SQRTLasso

General imports: 

```
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from matplotlib import pyplot
from numba import jit
from numba import njit
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
```

Building the sklearn-compliant function:

```
class SQRTLasso:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def fit(self, x, y):
        alpha=self.alpha
        def f_obj(x,y,beta,alpha):
          n =len(x)
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.sqrt(1/n*np.sum((y-x.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
          return output
        
        def f_grad(x,y,beta,alpha):
          n=x.shape[0]
          p=x.shape[1]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.array((-1/np.sqrt(n))*np.transpose(x).dot(y-x.dot(beta))/np.sqrt(np.sum((y-x.dot(beta))**2))+alpha*np.sign(beta)).flatten()
          return output
        
        def objective(beta):
          return(f_obj(x,y,beta,alpha))

        def gradient(beta):
          return(f_grad(x,y,beta,alpha))
        
        beta0 = np.ones((x.shape[1],1))
        output = minimize(objective, beta0, method='L-BFGS-B', jac=gradient,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
```

### SCAD (Smoothly Clipped Absolute Deviation)

Smoothly Clipped Absolute Deviation, or SCAD, is also a penalty that attempts to improve upon popular variable selection methods such as Lasso. 
Specifically, it tackles the issue of bias. The penalty was originally introduced to encourage sparse solutions to the lease squares prtoblem while 
also allowing for large balues of beta. Unlike Lasso which increases its penalty as the absolute value of beta increases, SCAD penalizes based on the sign 
function of beta.

#### Building an sklearn-compliant implementation of SCAD


```
@njit
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part

@njit    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
```

The sklearn-compliant function
```
class SCAD(BaseEstimator, RegressorMixin):
    def __init__(self, a=2,lam=1):
        self.a, self.lam = a, lam
  
    def fit(self, x, y):
        a = self.a
        lam   = self.lam

        @njit
        def scad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          return 1/n*np.sum((y-x.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))

        @njit  
        def dscad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          output = -2/n*np.transpose(x).dot(y-x.dot(beta))+scad_derivative(beta,lam,a)
          return output.flatten()
        
        
        beta0 = np.zeros(p)
        output = minimize(scad, beta0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 50,'disp': False})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
```

## Simulating Data

We want to simulate 100 datasets, each with 1200 features, 200 observations, and a toeplitz correlation structure. 

```
n = 200 
p = 1200

beta_star = np.concatenate(([1]*7, [0]*25, [.25]*5, [0]*50, [0.7]*15, [0]*1098))

v = []
for i in range(p):
  v.append(0.8**i)

mu = [0]*p
sigma = 3.5

np.random.seed(123)
x = np.random.multivariate_normal(mu, toeplitz(v), size=n)
y = np.matmul(x,beta_star) + sigma*np.random.normal(0,1,n)
```

## Implementing Regularization Methods 

### SQRTLasso

Using GridSearchCV, we will first calculate the optimal parameters for the model. 

```
grid = GridSearchCV(estimator=SQRTLasso(),cv=5,scoring='neg_mean_squared_error',param_grid={'alpha': np.linspace(0.001, 1, 50)})
grid.fit(X, y)
```

We will calculate three metrics to measure the performance: number of true non-zero coefficients, L2 distance to the ideal solution, and Root Mean Square Error.

First, we fit the model. 

```
model = SQRTLasso(alpha = grid.best_params_.get('alpha'))
model.fit(X,y)
```
Calculate true non-zero coefficients:

```
pos = np.where(beta_star != 0)
beta_hat = model.coef_
pos_sqrtlasso = np.where(beta_hat != 0)
print(len(np.intersect1d(pos, pos_sqrtlasso)))
```

We find that there were 27 true non-zero coefficients. 

L2 distance to ideal solution:

```
print(np.linalg.norm(model.coef_-beta_star,ord=2))
```

L2 distance is 1.1709479610976854.

RMSE:

```
yhat = model.predict(X)
print((MSE(yhat, y)**0.5))
```

The resulting RMSE was 3.4487527033369685.

### SCAD 

Like we did previously, we will use GridSearchCV to define the ideal model parameters. 

```
grid = GridSearchCV(estimator=SCAD(),cv=5,scoring='neg_mean_squared_error',param_grid=[{'lam': np.linspace(0.001, 1, 20), 'a': np.linspace(0.1, 3, 20)}])
grid.fit(X, y)
```

Now we will calculate intersction with ground truth, L2 distace, and RMSE

```
model = SCAD(lam = grid.best_params_.get('lam'), a = grid.best_params_.get('a'))
model.fit(X,y)

# finding intersection with ground truth 
pos = np.where(beta_star != 0)
beta_hat = model.coef_
pos_sqrtlasso = np.where(beta_hat != 0)
print(len(np.intersect1d(pos, pos_sqrtlasso)))

# calculating L2 Dist
print(np.linalg.norm(model.coef_-beta_star,ord=2))

# finding RMSE
yhat = model.predict(X)
print((MSE(yhat, y)**0.5))
```

True nonzero coefficients: 27
L2 distance to the ideal solution: 3.0179973698451623
RMSE: 0.007254548726807134

### 










