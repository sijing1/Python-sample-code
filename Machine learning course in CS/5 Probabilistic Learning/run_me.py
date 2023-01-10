#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv

import matplotlib.pyplot as plt
from scipy.stats import norm


# ## Question 1

# In[2]:


def prior(m):
    if m == 0:
        return 0.25
    elif m == 1 or m == 2:
        return 0.2
    elif m == 3 or m == 4:
        return 0.1
    elif m == 5:
        return 0.05
    elif m >= 6:
        return 0.025
    return 0


# ## Question 2

# In[3]:


x = range(10)
y = [prior(m) for m in x]

plt.bar(x, y)
plt.xlabel('m')
plt.ylabel('p(m)')
plt.show()


# ## Question 3

# In[4]:


def f_predictor(x, m):
    return x ** m if m > 0 else 0

def likelihood_single(x,y,m):
    mu = f_predictor(x, m)
    return norm.pdf(y, mu, 0.1)


# ## Question 4

# In[5]:


def likelihood(X,Y,m):
    p = [likelihood_single(X[i], Y[i], m) for i in range(10)]
    result = np.prod(p)
    return result


# ## Question 5

# In[6]:


X = np.loadtxt('x.csv')
Y = np.loadtxt('y.csv')

x_axis = range(10)
y_axis = [likelihood(X, Y, i) for i in x_axis]

plt.bar(x_axis, y_axis)
plt.xlabel('m')
plt.ylabel('likelihood')
plt.show()


# ## Question 7

# In[7]:


def posterior(X,Y,m):
    llh = likelihood(X, Y, m)
    p_data = np.sum([likelihood(X, Y, i) * prior(i) for i in range(10)])
    return prior(m) * llh / p_data


# ## Question 8

# In[8]:


X = np.loadtxt('x.csv')
Y = np.loadtxt('y.csv')

x_axis = range(10)
y_axis = [posterior(X, Y, i) for i in x_axis]

plt.bar(x_axis, y_axis)
plt.xlabel('m')
plt.ylabel('posterior')
plt.show()


# In[9]:


# Make sure Q8 is correct. It should be really close to 1 (giving it some range of errors).
assert (1 - np.sum(y_axis)) < 0.0001


# ## Question 9

# In[10]:


def MAP(X, Y):
    candidates = [posterior(X, Y, i) for i in range(10)]
    return np.argmax(candidates)


# ## Question 10

# In[11]:


m = MAP(X, Y)
print('MAP:', m)
print('Posterior Probability:', [posterior(X,Y,i) for i in range(10)][m])


# ## Question 11

# In[12]:


def predict_MAP(x,X,Y):
    m = MAP(X, Y)
    return f_predictor(x, m)


# ## Question 12

# In[13]:


X_test = np.loadtxt('x_test.csv')
Y_test = np.loadtxt('y_test.csv')


# In[14]:


y_pred = [predict_MAP(x, X, Y) for x in X_test]

plt.scatter(X_test, Y_test, marker="x", label="y_true")
plt.scatter(X_test, y_pred, marker="o", label="y_predict")
plt.xlabel('x-test')
plt.ylabel('y')
plt.legend()
plt.show()


# ## Question 13

# In[15]:


MSE = np.mean((y_pred - Y_test) ** 2)
print("MSE:", MSE)


# ## Question 14

# In[16]:


def predict_Bayes(x,X,Y):
    f = np.sum([posterior(X, Y, i) * f_predictor(x, i) for i in range(10)])
    return f


# ## Question 15

# In[17]:


y_pred_bayes = [predict_Bayes(x, X, Y) for x in X_test]

plt.scatter(X_test, Y_test, marker="x", label="y_true")
plt.scatter(X_test, y_pred_bayes, marker="o", label="y_predict")
plt.xlabel('x-test')
plt.ylabel('y')
plt.legend()
plt.show()


# ## Question 16

# In[18]:


MSE = np.mean((y_pred_bayes - Y_test) ** 2)
print("MSE (Bayes):", MSE)

