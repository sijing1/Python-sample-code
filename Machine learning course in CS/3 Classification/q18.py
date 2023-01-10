#!/usr/bin/env python
# coding: utf-8

# In[2]:


import autograd, autograd.misc
from autograd import numpy as np

import csv
from matplotlib import pyplot as plt

from sklearn.model_selection import KFold


# In[3]:


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


# In[4]:


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


# ## Q15

# In[5]:


def prediction_loss_full(X,Y,W,V,b,c,λ):
    W_X = X @ W.T  # (N, M)
    f_X = np.dot(np.tanh(W_X + b), V.T) + c

    # I don't have a better way to do this operation with only np functions.
    f_y_X = -f_X[np.arange(f_X.shape[0]), Y]  # (N,)
    
    # Calculate the mix of logistic loss.
    L = np.sum(np.array(f_y_X) + np.log(np.sum(np.exp(f_X), axis=1)))

    # Calculate the regularization.
    r = λ * (np.sum(W ** 2) + np.sum(V ** 2))
    
    return L + r


# ## Q16

# In[6]:


def prediction_grad_full(X,Y,W,V,b,c,λ):
    grad_func = autograd.grad(prediction_loss_full, [2, 3, 4, 5])
    dLdW, dLdV, dLdb, dLdc = grad_func(X,Y,W,V,b,c,λ)
    return dLdW, dLdV, dLdb, dLdc


# ## Q17

# In[7]:


# Hyperparameters

step_size = .000025
max_iter = 1000
momentum = 0.1
λ = 1.0

verbose_count = 10  # This is how many progress should be printed.


# In[8]:


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

# In[9]:


hidden_layer_size_candidates = [5, 40, 70]


# ## Q18

# In[17]:


best_hidden_layer_size = None
best_acc = np.inf


# In[18]:


# Data preparation.
kf = KFold(n_splits=2)
train_index, test_index = list(kf.split(X_trn_reshape))[0]
# Extract the training and testing set by using the k-fold result.
X_train, X_test = X_trn_reshape[train_index], X_trn_reshape[test_index]
y_train, y_test = y_trn[train_index], y_trn[test_index]


# In[19]:


# Huge works.
for hidden_layer_size in hidden_layer_size_candidates:
    print('>>>>> hidden layer size:', hidden_layer_size)
    nn = NeuralNetwork(hidden_layer_size)
    nn.fit(X_train, y_train)
    f_X = nn.decision_function(X_test)
    curr_val_loss = compute_logistic_loss(f_X, y_test)
    print('final logistic loss:', curr_val_loss)
    y_pred = np.argmax(f_X, axis=1)
    curr_acc = np.mean(y_pred == y_test)
    print('accuracy:', curr_acc)
    if curr_acc > best_acc:
        best_hidden_layer_size = hidden_layer_size
        best_acc = curr_acc


# In[20]:


best_hidden_layer_size


# In[16]:


best_acc


# ### Q18 Finalized Training

# In[14]:


# Finalized training.
nn_finalized = NeuralNetwork(best_hidden_layer_size)
nn_finalized.fit(X_trn_reshape, y_trn)


# In[15]:


f_X = nn_finalized.decision_function(X_tst_reshape)
y_pred = np.argmax(f_X, axis=1)
write_csv(y_pred, "Q18-Lingxi.csv")

