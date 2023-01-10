#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ## Load the image

# In[2]:


img = Image.open("HCM.png")
img.show()
px = img.load()
width, height = img.size
pixs = np.empty([width, height, 3])
for w in range(width):
    for h in range(height):
        coordinate = w, h
        cur = img.getpixel(coordinate)
        pixs[w, h] = cur[:3]


# ## Question 3

# In[3]:


# Build chunks.
pixs_chunks = []
for j in range(0, height, 3):
    for i in range(0, width, 3):
        curr_block_elements = []
        for j_shift in range(3):
            for i_shift in range(3):
                curr_block_elements.append(pixs[i + i_shift, j + j_shift])
        pixs_chunks.append(np.concatenate(curr_block_elements))


# In[4]:


K = [2, 5, 10, 25, 50, 100, 200, 1000]
new_img_dict = dict()
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixs_chunks)
    labels = kmeans.labels_
    curr_image = np.empty([width, height, 3])
    for i in range(len(labels)):
        label = labels[i]
        chunks = kmeans.cluster_centers_[label].reshape(9, 3)
        base_x, base_y = (i % 150) * 3, i // 150 * 3
        for y_shift in range(3):
            for x_shift in range(3):
                curr_image[base_x + x_shift, base_y + y_shift] = chunks[3 * y_shift + x_shift]
    print('k:', k)
    img = Image.fromarray(curr_image.swapaxes(0, 1).astype(np.uint8), 'RGB')
    img.show()

    new_img_dict[k] = curr_image


# ## Question 4

# In[5]:


for k in K:
    print(k, '---', np.mean((new_img_dict[k] - pixs) ** 2))


# ## Question 5

# In[6]:


for k in K:
    print(k, '---', width * height / 9 + 27 * k)


# ## Question 6

# In[7]:


for k in K:
    print(k, '---', (width * height / 9 + 27 * k) / (width * height * 3))


# ## Question 11

# In[8]:


x_sum = np.zeros([2500, 1])
img_lst = []
for i in range(100):
    img_name = 'face_' + str(i) + '.png'
    img = np.array(mpimg.imread("Faces/" + img_name)).reshape(2500, 1)
    img_lst.append(img)
    x_sum += img
x_mean = x_sum / 100
num_img = len(img_lst)
C = np.zeros([2500, 2500])
for i in range(100):
    img = img_lst[i]
    diff = np.array(img - x_mean)
    C += np.dot(diff, diff.T)
C = C / num_img
U, s, VT = np.linalg.svd(C, full_matrices=True)


# In[9]:


K = [3, 5, 10, 30, 50, 100]
face_img = np.array(mpimg.imread("face.png")).reshape(2500, 1)
for k in K:
    reconstruct = np.zeros(shape=(1, 2500))
    for i in range(k):
        compressed_version = np.dot(U.T[i,], face_img)
        reconstruct = np.add(reconstruct, U.T[i,] * compressed_version)
    img = reconstruct.reshape(50, 50)
    plt.figure()
    print('k:', k)
    plt.imshow(img, cmap='gray')


# ## Question 12

# In[10]:


for k in K:
    print('k:', k, '---', (k * 2500 + 100 * k) / (100 * 2500))

