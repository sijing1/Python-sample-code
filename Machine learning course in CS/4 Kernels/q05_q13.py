#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt

stuff=np.load("data_synth.npz")
X_trn = stuff['X_trn']
Y_trn = stuff['y_trn']
X_val = stuff['X_val']
Y_val = stuff['y_val']

def get_poly_expansion(P):
    def expand(X):
        tmp = [np.sqrt(comb(P,p))*X**p for p in range(P+1)]
        return np.vstack(tmp).T
    return expand

# example usage
# h = get_poly_expansion(5)
# expansion = h(X_trn[0])
print(X_trn.shape)


# In[2]:


# Question 5 
def eval_basis_expanded_ridge(x,w,h):
    expansion=h(x)
    y = np.dot(w.T, expansion.T)
    return float(y)


# In[3]:


# Question 6
def train_basis_expanded_ridge(X,Y,λ,h):
    H=[]
    for i in range(len(X)):
        h_x = list(h(X[i])[0])
        H.append(h_x)
    H = np.matrix(H)
    A = np.dot(H.T, H) + λ * np.identity(len(H[0]))
    B = np.dot(H.T, Y)
    w = np.linalg.solve(A, B.T)
    return w


# In[4]:


# Question 7
P_candidate=[1,2,3,5,10]
λ=0.1
W=[]
for p in P_candidate:
    h = get_poly_expansion(p)
    w = train_basis_expanded_ridge(X_trn,Y_trn,λ,h)
    W.append(w)
    print(p,w)
    Y_pre=[]
    for i in range(len(X_trn)):
        x=X_trn[i]
        y_pre= (x,w,h)
        Y_pre.append(y_pre)
    str_label='p='+str(p)
    plt.scatter(X_trn, Y_trn, label='y_trn')
    plt.scatter(X_trn,Y_pre, label='y_pred')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y_pre')
    plt.title(str_label)
    plt.show()


# In[5]:


# Question 8
def get_poly_kernel(P):
    # x --> [[x]]
    def k(x,xp):
        x = np.array([x])
        xp = np.array([xp])
        kernel_value = np.power(np.dot(x, xp) + 1, P)
        return kernel_value
    return k


# In[6]:


# Question 9
x  = 0.1
xp = 0.5

k  = get_poly_kernel(5)
h  = get_poly_expansion(5)
out1 = k(x,xp)
out2 = np.inner(h(x),h(xp))
print("output 1", out1)
print("output 2", out2)


# In[7]:


# Question 10
def train_kernel_ridge(X,Y,λ,k):
    N = X.shape[0]
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i][j] = k(X[i], X[j])
    A = K + λ * np.identity(N)
    α = np.linalg.solve(A, Y)
    return α


# In[8]:


# Question 11
def eval_kernel_ridge(X_trn, x, α, k):
    y = 0
    N = X_trn.shape[0]
    for i in range(N):
        y += np.dot(α[i], k(X_trn[i], x))
    return y


# In[9]:


# Question 12
P = [1, 2, 3, 5, 10]
λ = 0.1
for p in P:
    k = get_poly_kernel(p)
    α = train_kernel_ridge(X_trn, Y_trn, λ, k)
    y_pred = []
    for x in X_trn:
        y = eval_kernel_ridge(X_trn, x, α, k)
        y_pred.append(y)
    str_label='p='+str(p)
    plt.scatter(X_trn, Y_trn, label='y_trn')
    plt.scatter(X_trn,y_pred, label='y_pred')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y_pre')
    plt.title(str_label)
    plt.show()


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=39e0fd1d-6f49-4b9e-b070-ceffe9a13f8f' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
