#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import csv
from sklearn import tree
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def show(x):
    img = x.reshape((3,31,31)).transpose(1,2,0)
    plt.imshow(img)
    plt.axis('off')
    plt.draw()
    plt.pause(0.01)
    

def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y])


stuff = np.load("data.npz")
X_trn = stuff["X_trn"]
y_trn = stuff["y_trn"]
X_tst = stuff["X_tst"]

X_trn_shape = X_trn.shape
X_trn_reshape = X_trn.reshape((X_trn_shape[0], X_trn_shape[1]*X_trn_shape[2]))

X_tst_shape = X_tst.shape
X_tst_reshape = X_tst.reshape((X_tst_shape[0], X_tst_shape[1]*X_tst_shape[2]))


# In[3]:


lam = 100
max_iter = 10000


# In[4]:


log_reg = LogisticRegression(
    tol=1e-4,
    C=(1/lam),
    max_iter=max_iter,
    # verbose=1,
    n_jobs=-1,
)

log_reg.fit(X_trn_reshape, y_trn)


# In[5]:


y_pred = log_reg.predict(X_tst_reshape)


# In[6]:


write_csv(y_pred, "Q10-Lingxi.csv")

