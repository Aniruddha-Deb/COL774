#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import pickle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


class SoftSVM_QP:
    
    def __init__(self, C=1):
        self.C = C
        
    def _qp_svm(self, P, q, G, h, A, b, X, y):
        self.sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(self.sol['x'])
        sv_idxs = (alphas > 1e-6).flatten()
        self.sv = X[sv_idxs]
        self.sv_y = y[sv_idxs]
        self.alphas = alphas[sv_idxs]
        
        self.b = self.sv_y - np.sum((self.sv @ self.sv.T) * self.alphas * self.sv_y, axis=0)
        self.b = np.mean(self.b)
    
    def fast_fit(self, X, y):
        # uses cupy + gpu acceleration
        import cupy
        X = X.astype(np.double)
        y = y.astype(np.double)
        Xc, yc = cupy.array(X), cupy.array(y)
        Pc = (yc @ yc.T) * (Xc @ Xc.T)
        m, n = X.shape
        P = matrix(cupy.asnumpy(Pc), tc='d')
        q = matrix(-np.ones((m, 1)), tc='d')
        A = matrix(y.T, tc='d')
        b = matrix(np.zeros(1), tc='d')
        G = matrix(np.vstack([-np.eye(m), np.eye(m)]), tc='d')
        h = matrix(np.hstack([np.zeros(m), self.C*np.ones(m)]).reshape(-1,1), tc='d')

        self._qp_svm(P, q, G, h, A, b, X, y)
    
    # X, y need to be numpy arrays
    def fit(self, X, y):
        m, n = X.shape
        X = X.astype(np.double)
        y = y.astype(np.double)
        p_np = (y @ y.T) * (X @ X.T)
        P = matrix(p_np, tc='d') # half not required
        q = matrix(-np.ones((m, 1)), tc='d')
        A = matrix(y.T, tc='d')
        b = matrix(np.zeros(1), tc='d')
        G = matrix(np.vstack([-np.eye(m), np.eye(m)]), tc='d')
        h = matrix(np.hstack([np.zeros(m), self.C*np.ones(m)]).reshape(-1,1), tc='d')

        self._qp_svm(P, q, G, h, A, b, X, y)

    def predict(self, X):
        prod = np.sum((self.sv @ X.T) * self.alphas * self.sv_y, axis=0) + self.b
        return np.sign(prod)


# ## Testing

# In[5]:


train_data_file = open('/kaggle/input/col774a2q2data/part2_data/train_data.pickle', 'rb')
train_data = pickle.load(train_data_file)
train_data_file.close()

# ah, have to use (d) and (d+1) mod 5
s = train_data['data'].shape
train_data['data'] = train_data['data'].reshape(s[0], np.product(s[1:]))

data_idx = ((train_data['labels'] == (9%5)) | (train_data['labels'] == ((9+1)%5))).flatten()
train_data['data'] = train_data['data'][data_idx,:]
train_data['labels'] = train_data['labels'][data_idx]
train_data['labels'] = np.where(train_data['labels'] == (9%5), 1, -1)



model = SoftSVM_QP()
model.fit(train_data['data'], train_data['labels'])


# In[ ]:


test_data_file = open('/kaggle/input/col774a2q2data/part2_data/test_data.pickle', 'rb')
test_data = pickle.load(test_data_file)
test_data_file.close()

s = test_data['data'].shape
test_data['data'] = test_data['data'].reshape(s[0], np.product(s[1:]))

data_idx = ((test_data['labels'] == (9%5)) | (test_data['labels'] == ((9+1)%5))).flatten()
test_data['data'] = test_data['data'][data_idx,:]
test_data['labels'] = test_data['labels'][data_idx]
test_data['labels'] = np.where(test_data['labels'] == (9%5), 1, -1)


# In[ ]:


preds = model.predict(test_data['data'])


# In[ ]:


count = np.count_nonzero(preds == test_data['labels'].flatten())
acc = count/test_data['labels'].size
print(f"Accuracy: {count}/{test_data['labels'].size} = {acc}")


# In[ ]:





# In[ ]:




