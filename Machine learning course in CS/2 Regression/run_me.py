"""CS 589 Assignment 2.ipynb
"""

import numpy as np

stuff = np.load("data.npz")
X_trn = stuff["X_trn"]
y_trn = stuff["y_trn"]
X_val = stuff["X_val"]
y_val = stuff["y_val"]

def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(np.subtract(x, y))))

def print_table(arr, leading_header=False):
    # Prof does not allow us to use pandas.
    # So I implement this for rounding values in matrix basis.
    for line in arr:
        for i in range(len(line)):
            if leading_header and i == 0:
                line[i] = '{:<8}'.format(line[i])
            else:
                line[i] = '{:10.6f}'.format(round(line[i], 6))
        print('  '.join(line))

"""# Question 1"""

def KNN_reg_predict(X_trn, y_trn, x, K):
    neighbors = []
    for i in range(len(X_trn)):
        neighbors.append((i, euclidean_distance(x, X_trn[i])))
    neighbors = sorted(neighbors, key=lambda item: item[1])[:K]
    return sum([y_trn[neighbor[0]] for neighbor in neighbors]) / K

"""# Question 2

## With MSE
"""

output_table = []
for k in range(1, 11):
    # Compute training error.
    training_error = 0
    for i in range(len(X_trn)):
        y_predicted = KNN_reg_predict(X_trn, y_trn, X_trn[i], k)
        training_error += (y_predicted - y_trn[i]) ** 2
    training_error /= len(X_trn)

    # Compute testing error.
    testing_error = 0
    for i in range(len(X_val)):
        y_predicted = KNN_reg_predict(X_trn, y_trn, X_val[i], k)
        testing_error += (y_predicted - y_val[i]) ** 2
    testing_error /= len(X_val)

    output_table.append([training_error, testing_error])

print_table(output_table)

"""## With MAE"""

output_table = []
for k in range(1, 11):
    # Compute training error.
    training_error = 0
    for i in range(len(X_trn)):
        y_predicted = KNN_reg_predict(X_trn, y_trn, X_trn[i], k)
        training_error += abs(y_predicted - y_trn[i])
    training_error /= len(X_trn)

    # Compute testing error.
    testing_error = 0
    for i in range(len(X_val)):
        y_predicted = KNN_reg_predict(X_trn, y_trn, X_val[i], k)
        testing_error += abs(y_predicted - y_val[i])
    testing_error /= len(X_val)

    output_table.append([training_error, testing_error])

print_table(output_table)

"""# Question 3"""

def linear_reg_predict(x, w):
    y = np.dot(w.T, x)  # w^T x
    return y

"""# Question 4"""

def linear_reg_train(X_trn, y_trn, l):
    # I changed `np.eye` to `np.identity`.
    # They works the same but this one is what we really want.
    # https://numpy.org/doc/stable/reference/generated/numpy.identity.html
    A = np.dot(X_trn.T, X_trn) + l * np.identity(len(X_trn[0]))
    B = np.dot(X_trn.T, y_trn)
    w = np.linalg.solve(A, B)
    return w

"""# Question 5"""

l_list = [0, 0.001, 0.01, 0.1, 1, 10]
Re = []
for l in l_list:
    # train the model
    w = linear_reg_train(X_trn, y_trn, l)
    
    Pre_y_train=[]
    Pre_y_test=[]
    
    # compute train error
    for i in range(len(y_trn)):
        x_now = X_trn[i,:]
        pre_y_train = np.dot(w.T, x_now)
        Pre_y_train.append(pre_y_train)
    error_train = Pre_y_train - y_trn
    error_train_abs = sum([abs(x) for x in error_train]) / len(y_trn)
    error_train_mse = sum([x*x for x in error_train]) / len(y_trn)


    # compute test error
    for i in range(len(y_val)):
        x_now = X_val[i,:]
        pre_y_test = np.dot(w.T, x_now)
        Pre_y_test.append(pre_y_test)
    error_test = Pre_y_test - y_val
    error_test_abs = sum([abs(x) for x in error_test]) / len(y_val)
    error_test_mse = sum([x ** 2 for x in error_test]) / len(y_val)
    
    Re.append([l, error_train_mse, error_test_mse, error_train_abs, error_test_abs])

print_table(Re, leading_header=True)

"""# Question 7

"""

def reg_stump_predict(x, dim, thresh, c_left, c_right):
    if x[dim] <= thresh:
        return c_left
    else:
        return c_right

"""# Question 8

"""

def reg_stump_train(X_trn, y_trn):
    D = len(X_trn[0])  # Total available dimensions
    len_candidates = len(X_trn)
    smallest_err = np.inf  # Use infinity, so the first value will always be stored.
    dim = None
    thresh = None
    c_left = None
    c_right = None
    
    # Prepare the combo.
    X_y_trn = [(X_trn[i], y_trn[i]) for i in range(len_candidates)]
    
    for d in range(D):  # For each dimension.
        # Sort the training combos based on current dimension.
        X_y_ordered = sorted(X_y_trn, key=lambda x: x[0][d])
        # Pre-compute the candidates.
        # 0 means front (left) side of the split.
        # 1 means back (right) side of the split.
        sum_0 = 0
        sum_1 = np.sum([cand[1] for cand in X_y_ordered])
        list_0 = []  # This represents the list of left side.
        list_1 = X_y_ordered[:]  # This represents the list of right side.
        # Then, for each candidates in X_y_ordered.
        for i in range(len_candidates - 1):
            # Get the candidate that is going to be shifted.
            # We will shift it from right to left.
            cand_shift = X_y_ordered[i]
            # `cand` means candidates.
            # cand_shift[1] represents the label (y).
            sum_0 += cand_shift[1]
            sum_1 -= cand_shift[1]
            list_0 += [list_1[0]]
            list_1 = list_1[1:]
            c_0 = sum_0 / len(list_0)
            c_1 = sum_1 / len(list_1)
            # Compute MSE for both sides.
            mse_0 = np.sum([np.square(cand[1] - c_0) for cand in list_0])
            mse_1 = np.sum([np.square(cand[1] - c_1) for cand in list_1])
            err = mse_0 + mse_1
            # When this error is smaller than others, then we output this combo.
            if err < smallest_err:
                smallest_err = err
                # Compute and store the thresh only when it is the smallest.
                point_0 = X_y_ordered[i][0]
                point_1 = X_y_ordered[i+1][0]
                thresh = np.mean([point_0[d], point_1[d]])
                # Then store the rest of info.
                dim = d
                c_left = c_0
                c_right = c_1
    
    return dim, thresh, c_left, c_right

"""# Question 9"""

dim, thresh, c_left, c_right = reg_stump_train(X_trn, y_trn)

# Predict by training data.
predicted_y_trn = [reg_stump_predict(point, dim, thresh, c_left, c_right) for point in X_trn]

# Predict by test data.
predicted_y_val = [reg_stump_predict(point, dim, thresh, c_left, c_right) for point in X_val]

err_trn = predicted_y_trn - y_trn
err_val = predicted_y_val - y_val

mse_trn = np.mean(np.square(err_trn))
mae_trn = np.mean(np.absolute(err_trn))

mse_val = np.mean(np.square(err_val))
mae_val = np.mean(np.absolute(err_val))

print("Train MSE: ", mse_trn)
print("Train MAE: ", mae_trn)
print("Test MSE: ", mse_val)
print("Test MAE: ", mae_val)

"""# Real Data Section"""

real_data = np.load("big_data.npz")
X_trn_real = real_data["X_trn"]
y_trn_real = real_data["y_trn"]
X_val_real = real_data["X_val"]
y_val_real = real_data["y_val"]

"""# Question 10

"""

from sklearn.neighbors import KNeighborsRegressor

k_list = [1, 2, 5, 10, 20, 50]
train_error_lst = []
test_error_lst = []

for k in k_list:
  neigh = KNeighborsRegressor(n_neighbors = k)
  neigh.fit(X_trn_real, y_trn_real)

  train_y_predict = []
  for train_data in X_trn_real:
    train_y_predict.append(neigh.predict([train_data])[0])
  train_error = y_trn_real - train_y_predict
  train_error_MSE = sum([x*x for x in train_error]) / len(y_trn_real)
  train_error_lst.append(train_error_MSE)

  test_y_predict = []
  for test_data in X_val_real:
    test_y_predict.append(neigh.predict([test_data])[0])
  test_error = y_val_real - test_y_predict
  test_error_MSE = sum([x*x for x in test_error]) / len(y_val_real)
  test_error_lst.append(test_error_MSE)
print("Train Error: ", train_error_lst)
print("Test Error: ", test_error_lst)

"""# Question 11"""

from sklearn.tree import DecisionTreeRegressor

depth_list = [1, 2, 3, 4, 5]
train_error_lst = []
test_error_lst = []

for depth in depth_list:
  regressor = DecisionTreeRegressor(max_depth=depth)
  regressor.fit(X_trn_real, y_trn_real)
  
  train_y_predict = []
  for train_data in X_trn_real:
    train_y_predict.append(regressor.predict([train_data])[0])
  train_error = y_trn_real - train_y_predict
  train_error_MSE = sum([x*x for x in train_error]) / len(y_trn_real)
  train_error_lst.append(train_error_MSE)

  test_y_predict = []
  for test_data in X_val_real:
    test_y_predict.append(regressor.predict([test_data])[0])
  test_error = y_val_real - test_y_predict
  test_error_MSE = sum([x*x for x in test_error]) / len(y_val_real)
  test_error_lst.append(test_error_MSE)
print("Train Error: ", train_error_lst)
print("Test Error: ", test_error_lst)

"""# Question 12

"""

from sklearn.linear_model import Ridge
l_lst = [0, 1, 10, 100, 1000, 10000]
train_error_lst = []
test_error_lst = []

for l in l_lst:
  clf = Ridge(alpha = l)
  clf.fit(X_trn_real, y_trn_real)

  train_y_predict = []
  for train_data in X_trn_real:
    train_y_predict.append(clf.predict([train_data])[0])
  train_error = y_trn_real - train_y_predict
  train_error_MSE = sum([x*x for x in train_error]) / len(y_trn_real)
  train_error_lst.append(train_error_MSE)

  test_y_predict = []
  for test_data in X_val_real:
    test_y_predict.append(clf.predict([test_data])[0])
  test_error = y_val_real - test_y_predict
  test_error_MSE = sum([x*x for x in test_error]) / len(y_val_real)
  test_error_lst.append(test_error_MSE)
print("Train Error: ", train_error_lst)
print("Test Error: ", test_error_lst)

"""# Question 13"""

from sklearn import linear_model
l_lst = [0, 0.1, 1, 10, 100, 1000]
train_error_lst = []
test_error_lst = []

for l in l_lst:
  lam = l / (2 * len(y_trn_real)) # Hint in question 13
  clf = linear_model.Lasso(alpha = lam)
  clf.fit(X_trn_real, y_trn_real)

  train_y_predict = []
  for train_data in X_trn_real:
    train_y_predict.append(clf.predict([train_data])[0])
  train_error = y_trn_real - train_y_predict
  train_error_MSE = sum([x*x for x in train_error]) / len(y_trn_real)
  train_error_lst.append(train_error_MSE)

  test_y_predict = []
  for test_data in X_val_real:
    test_y_predict.append(clf.predict([test_data])[0])
  test_error = y_val_real - test_y_predict
  test_error_MSE = sum([x*x for x in test_error]) / len(y_val_real)
  test_error_lst.append(test_error_MSE)
print("Train Error: ", train_error_lst)
print("Test Error: ", test_error_lst)