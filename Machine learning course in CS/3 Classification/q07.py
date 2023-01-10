#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
from sklearn import tree
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


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


# In[2]:


# Make sure dataset is great.
preview_index = 11
show(X_trn_reshape[preview_index])
print('label:', y_trn[preview_index])


# In[3]:


k_candidates = [1, 3, 5, 7, 9, 11]
mean_error = []

for k in k_candidates:
    knn_clf = KNeighborsClassifier(
        n_neighbors=k,
        n_jobs=-1,
    )
    error = 0
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X_trn_reshape):
        X_train, X_test = X_trn_reshape[train_index], X_trn_reshape[test_index]
        y_train, y_test = y_trn[train_index], y_trn[test_index]
        
        knn_clf = knn_clf.fit(X_train, y_train)
        y_pred = knn_clf.predict(X_test)
        cur_error = 1 - metrics.accuracy_score(y_test, y_pred)
        error += cur_error

    mean_error.append(error/5)


# In[4]:


# Print in a table-friendly manner
for i in range(len(mean_error)):
    print(k_candidates[i], ':', round(mean_error[i], 4))

