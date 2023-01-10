#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# ### Question 1

# To verify the hand-written results (by Xuming) of Q1.

# In[2]:


np.log(0.4)


# In[3]:


-(0.4 * np.log(0.4) + 0.6 * np.log(0.6))


# In[4]:


def cross_entropy(p: list):
    if not len(p):
        return 0
    total_item = len(p)
    p_1 = len([i for i in p if i[1] == 0]) / total_item
    p_2 = len([i for i in p if i[1] == 1]) / total_item
    print('p_1:', p_1, '| p_2:', p_2)
    log_1 = math.log(p_1) if p_1 else 0
    log_2 = math.log(p_2) if p_2 else 0
    return -1 * (p_1 * log_1 + p_2 * log_2)


# In[5]:


candidates = [
    (1, 1),
    (2, 1),
    (3, 0),
    (4, 0),
    (5, 1),
]

splits = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]


# In[6]:


i_p = cross_entropy(candidates)
print('I(p) =', i_p)
print()

for split_point in splits:
    cand_1 = [i for i in candidates if i[0] < split_point]
    cand_2 = [i for i in candidates if i[0] >= split_point]
    n_1 = len(cand_1)
    n_2 = len(cand_2)
    i_p1 = cross_entropy(cand_1)
    i_p2 = cross_entropy(cand_2)
    print('n_1:', n_1, '| n_2:', n_2, '| i_p1:', i_p1, '| i_p2:', i_p2)
    information_gain = i_p - (n_1 * i_p1 + n_2 * i_p2) / (n_1 + n_2)
    print(f'>>> IG for split {split_point} is {information_gain}')
    print()

