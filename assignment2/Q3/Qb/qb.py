import numpy as np
import time
from sklearn import svm
import sys
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
        return prod

def eq_block(x, y):
    i = 0
    while i < len(x)-1 and y[i] == y[i+1]:
        i += 1
    return i

class MultiClassSoftKernelSVM_QP:
    
    def __init__(self, kernel, C=1.0):
        self.kernel = kernel
        self.C = C
        self.svms = {}
        
    def extract_binary_problem(X, y, i, j):
        idx = ((y == i) | (y == j)).flatten()
        X = X[idx,:]
        y = y[idx]
        y = np.where(y == j, 1, -1)      
        return (X,y)
        
    def fit(self, X, y):
        # get the number of distinct classes first
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        # now make a list of tuples, defining the models. (a,b) => -1->a, 1->b
        for i in range(len(self.classes)):
            for j in range(i+1,len(self.classes)):
                svm = SoftKernelSVM_QP(self.kernel, C=self.C)
                # transform y and x to make
                Xp, yp = MultiClassSoftKernelSVM_QP.extract_binary_problem(X, y, self.classes[i], self.classes[j])
                svm.fit(Xp, yp)
                svm.learn(Xp, yp)
                self.svms[(i,j)] = svm
    
    def predict(self, X):
        m, n = X.shape
        counts = np.zeros((m,self.n_classes))
        scores = np.zeros((m,self.n_classes))
        for (i,j) in self.svms:
            score = self.svms[(i,j)].predict(X)
            pred = np.sign(score).astype(np.int8)
            pred = (pred+1)>>1 # convert to (0,1) predictions
            counts[:,i] += (1-pred)
            counts[:,j] += pred
            scores[:,i] -= (1-pred)*score
            scores[:,j] += pred*score
        
        
        self.counts = counts
        self.scores = scores
        predictions = np.zeros(m)
        counts_args = np.argsort(-counts, axis=1)
        self.counts_args = counts_args
        for i in range(m):
            eqlen = eq_block(counts_args[i],counts[i])
            if eqlen != 0:
                a = counts_args[i]
                max_j = 0
                for j in range(1,eqlen+1):
                    if scores[i][a[j]] > scores[i][a[max_j]]:
                        max_j = j
                predictions[i] = self.classes[a[max_j]]
            else:
                predictions[i] = self.classes[counts_args[i][0]]
        
        return predictions

def load_data(dir, type):
    data_file = open(f'{dir}/{type}.pickle', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    s = data['data'].shape
    data['data'] = data['data'].reshape(s[0], np.product(s[1:]))

    data['data'] = data['data'].astype(np.double)
    data['labels'] = data['labels'].astype(np.double)

    data['data'] = np.interp(data['data'], (0,255), (0,1))
    
    return data

if __name__ == "__main__":
    train_data = load_data(sys.argv[1], 'train_data')
    test_data = load_data(sys.argv[2], 'test_data')

    tic = time.perf_counter()
    model = MultiClassSoftKernelSVM_QP(create_gaussian_kernel())
    model.fit(train_data['data'], train_data['labels'])
    toc = time.perf_counter()
    qp_t = toc-tic

    preds = model.predict(test_data['data'])
    count = np.count_nonzero(preds == test_data['labels'].flatten())
    print(f"QP Training time: {qp_t}")
    print(f"QP Accuracy: {count}/{test_data['labels'].size} = {count/test_data['labels'].size}")

    # Scikit
    tic = time.perf_counter()
    clf = svm.SVC()
    clf.fit(train_data['data'], train_data['labels'].flatten())
    toc = time.perf_counter()
    sk_t = toc-tic
    print(f"Scikit training time: {sk_t} s")

    preds_sk = model.predict(test_data['data'])
    count_sk = np.count_nonzero(preds_sk == test_data['labels'].flatten())
    print(f"Scikit Accuracy: {count_sk}/{preds_sk.size} = {count/preds_sk.size}")

    # pickle and save predictions for next part
    pickle.dump(preds_sk.astype(np.uint8), open('../Qc/preds_sk.pkl', 'wb'))
    pickle.dump(preds_sk.astype(np.uint8), open('../Qc/preds.pkl', 'wb'))
