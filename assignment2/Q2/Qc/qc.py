import numpy as np
import time
from sklearn import svm
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

def check_preds(model, dataset):
    preds = model.predict(dataset['data'])
    count = np.count_nonzero(preds == dataset['labels'].flatten())
    acc = count/test_data['labels'].size
    return (count,acc)

if __name__ == "__main__":

    train_data = load_data(sys.argv[1], 'train_data', 9)
    test_data = load_data(sys.argv[2], 'test_data', 9)

    tic = time.perf_counter()
    sk_rbf = svm.SVC()
    sk_rbf.fit(train_data['data'], train_data['labels'].flatten())
    toc = time.perf_counter()
    sk_rbf_time = toc-tic
    (count,acc) = check_preds(sk_rbf, test_data)

    print(f"Scikit RBF SVM learns in {sk_rbf_time} s")
    print(f"Scikit RBF SVM nSV = {len(sk_rbf.support_)}")
    print(f"Scikit RBF SVM Accuracy = {count}/{test_data['labels'].size} = {acc}")
    print("")

    tic = time.perf_counter()
    sk_lin = svm.SVC(kernel='linear')
    sk_lin.fit(train_data['data'], train_data['labels'].flatten())
    toc = time.perf_counter()
    sk_lin_time = toc-tic
    (count,acc) = check_preds(sk_lin, test_data)

    print(f"Scikit linear SVM learns in {sk_lin_time} s")
    print(f"Scikit linear SVM nSV = {len(sk_lin.support_)}")
    print(f"Scikit linear SVM Accuracy = {count}/{test_data['labels'].size} = {acc}")
    print("")

    tic = time.perf_counter()
    model_gauss = SoftKernelSVM_QP(create_gaussian_kernel())
    model_gauss.fit(train_data['data'], train_data['labels'])
    model_gauss.learn(train_data['data'], train_data['labels'])
    toc = time.perf_counter()
    gaussian_qp_time = toc-tic
    print(f"QP gaussian SVM learns in {gaussian_qp_time} s")

    tic = time.perf_counter()
    model_lin = SoftKernelSVM_QP(linear_kernel)
    model_lin.fit(train_data['data'], train_data['labels'])
    model_lin.learn(train_data['data'], train_data['labels'])
    toc = time.perf_counter()
    linear_qp_time = toc-tic
    print(f"QP linear SVM learns in {linear_qp_time} s")

    lin_sv = set(model_lin.sv_idxs.flatten())
    gauss_sv = set(model_gauss.sv_idxs.flatten())
    sk_lin_sv = set(sk_lin.support_.flatten())
    sk_rbf_sv = set(sk_rbf.support_.flatten())

    int_sk_rbf_lin = len(sk_rbf_sv.intersection(lin_sv))
    int_sk_rbf_gauss = len(sk_rbf_sv.intersection(gauss_sv))
    int_sk_lin_lin = len(sk_lin_sv.intersection(lin_sv))
    int_sk_lin_gauss = len(sk_lin_sv.intersection(gauss_sv))

    print(f"{int_sk_rbf_lin} vectors match in sk_rbf model and QP lin model")
    print(f"{int_sk_rbf_gauss} vectors match in  sk_rbf model and QP gauss model")
    print(f"{int_sk_lin_lin} vectors match in sk_lin model and QP lin model")
    print(f"{int_sk_lin_gauss} vectors match in  sk_lin model and QP gauss model")

    print(f"2-norm between sk_lin weight and QP lin weight: {np.linalg.norm(sk_lin.coef_ - model_lin.w)}")
    print(f"Diff b/w b's of sk_lin and QP lin: {sk_lin.coef0} - {model_lin.b} = {sk_lin.coef0-model_lin.b}")
