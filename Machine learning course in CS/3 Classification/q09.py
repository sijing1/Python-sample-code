#!/usr/bin/env python
# coding: utf-8

# In[1]:


# For multi-thread of LinearSVC.
from joblib import parallel_backend


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


# ### Loss functions

# In[3]:


def compute_01_loss(decision_result, y_test):
    return np.mean((decision_result != y_test).astype(int))


# In[4]:


def compute_hinge_loss(y_pred, y_test):
    hinge_losses = []

    for i in range(len(y_pred)):
        scores = y_pred[i]
        expected_label = y_test[i]
        expected_score = scores[expected_label]
        other_scores = np.delete(scores, expected_label)
        
        max_other_score = max(other_scores)
        max_diff = max_other_score - expected_score
        hinge_loss = max(0, 1 + max_diff)
        hinge_losses.append(hinge_loss)

    return np.mean(hinge_losses)


# In[5]:


def compute_logistic_loss(y_pred, y_test):
    logistic_losses = []

    for i in range(len(y_pred)):
        scores = y_pred[i]
        expected_label = y_test[i]
        expected_score = scores[expected_label]
        
        # Logistic loss is summing over all scores, not "other scores".
        logistic_loss = -expected_score + np.log(np.sum(np.exp(scores)))
        logistic_losses.append(logistic_loss)

    return np.mean(logistic_losses)


# ### Trainer functions

# In[6]:


max_iter = 10000


# In[7]:


def hinge_trainer(lam, X_train, y_train, X_test):
    hinge_clf = LinearSVC(
        loss='hinge',
        tol=1e-4,
        C=(1/lam),
        # verbose=1,
        max_iter=max_iter,
    )

    with parallel_backend('threading', n_jobs=-1):
        # Train the hinge loss SVC.
        hinge_clf.fit(X_train, y_train)

    y_pred = hinge_clf.decision_function(X_test)
    decision_result = np.argmax(y_pred, axis=1)

    return y_pred, decision_result


# In[8]:


def logistic_trainer(lam, X_train, y_train, X_test):
    log_reg = LogisticRegression(
        tol=1e-4,
        C=(1/lam),
        max_iter=max_iter,
        # verbose=1,
        n_jobs=-1,
    )

    log_reg.fit(X_train, y_train)

    y_pred = log_reg.decision_function(X_test)
    decision_result = np.argmax(y_pred, axis=1)

    return y_pred, decision_result


# ### Main functions

# In[9]:


def train(lam):
    kf = KFold(n_splits=5)
    hinge_01_losses = []
    hinge_hinge_losses = []
    hinge_logistic_losses = []
    logistic_01_losses = []
    logistic_hinge_losses = []
    logistic_logistic_losses = []

    for train_index, test_index in kf.split(X_trn_reshape):
        # Extract the training and testing set by using the k-fold result.
        X_train, X_test = X_trn_reshape[train_index], X_trn_reshape[test_index]
        y_train, y_test = y_trn[train_index], y_trn[test_index]

        # Train with linear svc (hinge loss).
        y_pred, decision_result = hinge_trainer(lam, X_train, y_train, X_test)

        # 0-1 Loss.
        hinge_01_losses.append(compute_01_loss(decision_result, y_test))

        # Hinge Loss.
        hinge_hinge_losses.append(compute_hinge_loss(y_pred, y_test))

        # Logistic Loss.
        hinge_logistic_losses.append(compute_logistic_loss(y_pred, y_test))

        # Train with logistic regression.
        y_pred, decision_result = logistic_trainer(lam, X_train, y_train, X_test)

        # 0-1 Loss.
        logistic_01_losses.append(compute_01_loss(decision_result, y_test))

        # Hinge Loss.
        logistic_hinge_losses.append(compute_hinge_loss(y_pred, y_test))

        # Logistic Loss.
        logistic_logistic_losses.append(compute_logistic_loss(y_pred, y_test))
    
    hinge_result = (np.mean(hinge_01_losses), np.mean(hinge_hinge_losses), np.mean(hinge_logistic_losses))
    logistic_result = (np.mean(logistic_01_losses), np.mean(logistic_hinge_losses), np.mean(logistic_logistic_losses))
    return hinge_result, logistic_result


# In[10]:


def single_run(lam):
    hinge_result, logistic_result = train(lam)
    print('for lambda =', lam)
    hinge_01, hinge_hinge, hinge_logistic = hinge_result
    print('Training by hinge loss:')
    print('# 0-1 loss:', hinge_01)
    print('# hinge loss:', hinge_hinge)
    print('# logistic loss:', hinge_logistic)
    logistic_01, logistic_hinge, logistic_logistic = logistic_result
    print('Training by logistic loss:')
    print('# 0-1 loss:', logistic_01)
    print('# hinge loss:', logistic_hinge)
    print('# logistic loss:', logistic_logistic)


# ### Huge running works

# In[11]:


single_run(0.0001)


# In[12]:


single_run(0.01)


# In[13]:


single_run(1)


# In[14]:


single_run(10)


# In[15]:


single_run(100)


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=282728b2-0634-4525-bb42-de3942e68b42' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
