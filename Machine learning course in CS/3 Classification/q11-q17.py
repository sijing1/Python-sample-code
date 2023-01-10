#!/usr/bin/env python
# coding: utf-8

# In[1]:


import autograd, autograd.misc
from autograd import numpy as np

import csv
from matplotlib import pyplot as plt


# In[2]:


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


# ## Q11

# In[4]:


def prediction_loss(x,y,W,V,b,c):
    # Evaluate
    W_x = np.dot(W, x)
    f_x = c + np.dot(V, np.tanh(b + W_x))

    # Calculate Loss
    L = -f_x[y] + np.log(np.sum(np.exp(f_x)))
    return L


# ## Q12

# In[7]:


def softmax_single(f):
    bottom_part = np.sum(np.exp(f))
    return np.exp(f) / bottom_part


def prediction_grad(x,y,W,V,b,c):
    e_hat_y = np.zeros(x.shape[0])
    e_hat_y[y] = 1

    W_x = np.dot(W, x)
    h = np.tanh(b + W_x)
    f_x = c + np.dot(V, h)
    dLdf = -e_hat_y + softmax_single(f_x) # R^O

    dLdc = dLdf # R^O

    #              R^O -> R^(Ox1)                 R^M --> R^(1xM)    
    dLdV = np.dot(dLdf.reshape(dLdf.shape[0], 1), h.reshape(1, h.shape[0])) # R^(OxM)

    derivative_activation = 1 - np.tanh(b + W_x) ** 2
    dLdb = derivative_activation * np.dot(V.T, dLdf) # R^M

    #              R^M -> R^(Mx1)                 R^D --> R^(1xD)    
    dLdW = np.dot(dLdb.reshape(dLdb.shape[0], 1), x.reshape(1, x.shape[0]))

    return dLdW, dLdV, dLdb, dLdc


# ## Q13

# In[8]:


# Q13 Sample Data

x = np.array([1, 2]).astype(np.float64)

y = 1

W = np.array([
    [0.5, -1],
    [-0.5, 1],
    [1, 0.5],
]).astype(np.float64)

V = np.array([
    [-1, -1, 1],
    [1, 1, 1],
]).astype(np.float64)

b = np.array([0, 0, 0]).astype(np.float64)

c = np.array([0, 0]).astype(np.float64)


# In[9]:


dLdW, dLdV, dLdb, dLdc = prediction_grad(x, y, W, V, b, c)
print('dLdW:')
print(dLdW)
print()
print('dLdV:')
print(dLdV)
print()
print('dLdb:', dLdb)
print()
print('dLdc:', dLdc)


# ## Q14

# In[10]:


def prediction_grad_autograd(x,y,W,V,b,c):
    grad_func = autograd.grad(prediction_loss, [2, 3, 4, 5])
    dLdW, dLdV, dLdb, dLdc = grad_func(x,y,W,V,b,c)
    return dLdW, dLdV, dLdb, dLdc


# In[11]:


dLdW, dLdV, dLdb, dLdc = prediction_grad_autograd(x, y, W, V, b, c)
print('dLdW:')
print(dLdW)
print()
print('dLdV:')
print(dLdV)
print()
print('dLdb:', dLdb)
print()
print('dLdc:', dLdc)


# ## Q15

# In[12]:


def prediction_loss_full(X,Y,W,V,b,c,λ):
    W_X = X @ W.T  # (N, M)
    f_X = np.dot(np.tanh(W_X + b), V.T) + c

    f_y_X = -f_X[np.arange(f_X.shape[0]), Y]  # (N,)
    
    # Calculate the mix of logistic loss.
    L = np.sum(np.array(f_y_X) + np.log(np.sum(np.exp(f_X), axis=1)))

    # Calculate the regularization.
    r = λ * (np.sum(W ** 2) + np.sum(V ** 2))
    
    return L + r


# ## Q16

# In[15]:


def prediction_grad_full(X,Y,W,V,b,c,λ):
    grad_func = autograd.grad(prediction_loss_full, [2, 3, 4, 5])
    dLdW, dLdV, dLdb, dLdc = grad_func(X,Y,W,V,b,c,λ)
    return dLdW, dLdV, dLdb, dLdc


# ## Q17

# In[17]:


# Hyperparameters

step_size = .000025
max_iter = 1000
momentum = 0.1
λ = 1.0

verbose_count = 10  # How many progress should be printed.


# In[18]:


class NeuralNetwork:
    def __init__(self, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size

    def fit(self, X, Y):
        hidden_layer_size = self.hidden_layer_size
        input_dimension = X[0].shape[0]
        label_dimension = 4  # np.unique(Y).shape[0]

        W = np.random.randn(hidden_layer_size, input_dimension) / np.sqrt(input_dimension)
        V = np.random.randn(label_dimension, hidden_layer_size) / np.sqrt(input_dimension)
        b = np.zeros(hidden_layer_size)
        c = np.zeros(label_dimension)
        
        progress_count = 0.0

        training_losses = []
        average_grad_W = 0.0
        average_grad_V = 0.0
        average_grad_b = 0.0
        average_grad_c = 0.0
        for i in range(max_iter):
            curr_loss = prediction_loss_full(X,Y,W,V,b,c,λ)
            training_losses.append(curr_loss)

            # Gradient logic.
            dLdW, dLdV, dLdb, dLdc = prediction_grad_full(X,Y,W,V,b,c,λ)
            
            average_grad_W = (1.0 - momentum) * average_grad_W + momentum * dLdW
            average_grad_V = (1.0 - momentum) * average_grad_V + momentum * dLdV
            average_grad_b = (1.0 - momentum) * average_grad_b + momentum * dLdb
            average_grad_c = (1.0 - momentum) * average_grad_c + momentum * dLdc

            W = W - step_size * average_grad_W
            V = V - step_size * average_grad_V
            b = b - step_size * average_grad_b
            c = c - step_size * average_grad_c

            progress_count += 1
            if progress_count % (max_iter / verbose_count) == 0:
                f_X = np.dot(np.tanh(X @ W.T + b), V.T) + c
                curr_train_loss = compute_logistic_loss(f_X, Y)
                print('[NN] Train:', progress_count // (max_iter / 100), '%', '-', curr_train_loss)

        self.losses = training_losses

        self.W = W
        self.V = V
        self.b = b
        self.c = c

    def decision_function(self, X):
        W = self.W
        V = self.V
        b = self.b
        c = self.c
        return np.dot(np.tanh(X @ W.T + b), V.T) + c

    def params(self):
        return self.W, self.V, self.b, self.c

    def get_losses(self):
        return self.losses


# ### NN Training

# In[19]:


hidden_layer_size_candidates = [5, 40, 70]


# In[20]:


import time

f_X_all = []
training_losses_all = []

for hidden_layer_size in hidden_layer_size_candidates:
    print('>>>>> hidden layer size:', hidden_layer_size)
    start_time = time.time()
    nn = NeuralNetwork(hidden_layer_size)
    nn.fit(X_trn_reshape, y_trn)
    f_X = nn.decision_function(X_trn_reshape)
    f_X_all.append((hidden_layer_size, f_X))
    training_losses_all.append((hidden_layer_size, nn.get_losses()))
    end_time = time.time()
    print('running time:', round((end_time - start_time) * 1000))


# ### Loss Plot

# In[30]:


import matplotlib.pyplot as plt

training_losses_all
x = range(max_iter)
y_5 = training_losses_all[0][1]
y_40 = training_losses_all[1][1]
y_70 = training_losses_all[2][1]

plt.plot(x, y_5, label = 'Hidden_Layer = 5')
plt.plot(x, y_40, label = 'Hidden_Layer = 40')
plt.plot(x, y_70, label = 'Hidden_Layer = 70')
plt.xlabel('x - iter')
plt.ylabel('y - losses')
plt.legend()
plt.title('Q17 - Plot')
plt.savefig('Q17_Plot')

