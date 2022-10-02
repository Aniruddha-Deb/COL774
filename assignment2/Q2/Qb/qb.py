import numpy as np
import sys
from PIL import Image
from cvxopt import matrix, solvers
import pickle

def create_gaussian_kernel(gamma=0.001):
    def kernel(x, z):
        xx = (x**2).sum(axis=1).reshape(-1,1)@np.ones((1,z.shape[0]))
        zz = (z**2).sum(axis=1).reshape(-1,1)@np.ones((1,x.shape[0]))
        return np.exp(-gamma*(xx + zz.T - 2*x@z.T))
    
    return kernel

def linear_kernel(x, z):
    return x@z.T
      
class SoftKernelSVM_QP:
    
    def __init__(self, kernel, C=1):
        self.C = C
        self.kernel = kernel
        
    def learn(self, X, y, thresh=1e-6):
        alphas = np.array(self.sol['x'])
        idx = (alphas > thresh).flatten()
        self.sv_idxs = idx.nonzero()[0]
        print(f"Got {len(self.sv_idxs)} support vectors")
        self.sv = X[idx]
        self.sv_y = y[idx]
        self.alphas = alphas[idx]

        if (self.kernel == linear_kernel):
            self.w = self.sv.T @ (self.alphas * self.sv_y)
        
        self.b = self.sv_y - np.sum(self.kernel(self.sv,self.sv) * self.alphas * self.sv_y, axis=0)
        self.b = np.mean(self.b)
    
   # X, y need to be numpy arrays
    def fit(self, X, y):
        m, n = X.shape
        p_np = (y @ y.T) * self.kernel(X,X)
        P = matrix(p_np, tc='d') # half not required
        q = matrix(-np.ones((m, 1)), tc='d')
        A = matrix(y.T, tc='d')
        b = matrix(np.zeros(1), tc='d')
        G = matrix(np.vstack([-np.eye(m), np.eye(m)]), tc='d')
        h = matrix(np.hstack([np.zeros(m), self.C*np.ones(m)]).reshape(-1,1), tc='d')
        
        self.sol = solvers.qp(P, q, G, h, A, b)

    def predict(self, X):
        prod = np.sum(self.kernel(self.sv,X) * self.alphas * self.sv_y, axis=0) + self.b
        return np.sign(prod).astype(np.double)

def load_data(dir, type, digit):
    data_file = open(f'{dir}/{type}.pickle', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    s = data['data'].shape
    data['data'] = data['data'].reshape(s[0], np.product(s[1:]))

    data_idx = ((data['labels'] == (digit%5)) | (data['labels'] == ((digit+1)%5))).flatten()
    data['data'] = data['data'][data_idx,:]
    data['labels'] = data['labels'][data_idx]
    data['labels'] = np.where(data['labels'] == (digit%5), 1, -1)
    
    data['data'] = data['data'].astype(np.double)
    data['labels'] = data['labels'].astype(np.double)

    data['data'] = np.interp(data['data'], (0,255), (0,1))
    
    return data

if __name__ == "__main__":

    train_data = load_data(sys.argv[1], 'train_data', 9)
    test_data = load_data(sys.argv[2], 'test_data', 9)

    model = SoftKernelSVM_QP(create_gaussian_kernel())
    model.fit(train_data['data'], train_data['labels'])
    model.learn(train_data['data'], train_data['labels'])

    preds = model.predict(test_data['data'])

    count = np.count_nonzero(preds == test_data['labels'].flatten())
    acc = count/test_data['labels'].size
    print(f"Accuracy: {count}/{test_data['labels'].size} = {acc}")
    
    # Compare overlap with SV's of the other one
    model_lin = SoftKernelSVM_QP(linear_kernel)
    model_lin.fit(train_data['data'], train_data['labels'])
    model_lin.learn(train_data['data'], train_data['labels'])
    
    lin_sv = set(model_lin.sv_idxs.flatten())
    gauss_sv = set(model.sv_idxs.flatten())
    print(f"{len(lin_sv.intersection(gauss_sv))} vectors match in linear/gaussian SVM's")

    vecs = [(255*model.sv[i]).reshape(32,32,3).astype(np.uint8) for i in np.argsort(model.alphas.flatten())[:5]]
    for (i,vec) in enumerate(vecs):
        img = Image.fromarray(vec)
        img = img.resize((320,320), resample=Image.NEAREST)
        img.save(f'sv_{i+1}.png')
