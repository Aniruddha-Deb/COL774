import numpy as np
import sys
from PIL import Image
from cvxopt import matrix, solvers
import pickle

class SoftSVM_QP:
    
    def __init__(self, C=1):
        self.C = C
        
    def learn(self, X, y, thresh=1e-6):
        alphas = np.array(self.sol['x'])
        sv_idxs = (alphas > thresh).flatten()
        print(f"Got {np.count_nonzero(sv_idxs)} support vectors")
        self.sv = X[sv_idxs]
        self.sv_y = y[sv_idxs]
        self.alphas = alphas[sv_idxs]
        
        self.w = self.sv.T @ (self.alphas * self.sv_y)
        self.b = self.sv_y - np.sum((self.sv @ self.sv.T) * self.alphas * self.sv_y, axis=0)
        self.b = np.mean(self.b)
    
   # X, y need to be numpy arrays
    def fit(self, X, y):
        m, n = X.shape
        p_np = (y @ y.T) * (X @ X.T)
        P = matrix(p_np, tc='d') # half not required
        q = matrix(-np.ones((m, 1)), tc='d')
        A = matrix(y.T, tc='d')
        b = matrix(np.zeros(1), tc='d')
        G = matrix(np.vstack([-np.eye(m), np.eye(m)]), tc='d')
        h = matrix(np.hstack([np.zeros(m), self.C*np.ones(m)]).reshape(-1,1), tc='d')
        
        self.sol = solvers.qp(P, q, G, h, A, b)

    def predict(self, X):
        prod = X@self.w + self.b
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

    model = SoftSVM_QP()
    model.fit(train_data['data'], train_data['labels'])
    model.learn(train_data['data'], train_data['labels'])

    preds = model.predict(test_data['data'])

    count = np.count_nonzero(preds == test_data['labels'])
    acc = count/test_data['labels'].size
    print(f"Accuracy: {count}/{test_data['labels'].size} = {acc}")

    vecs = [(255*model.sv[i]).reshape(32,32,3).astype(np.uint8) for i in np.argsort(model.alphas.flatten())[:5]]
    for (i,vec) in enumerate(vecs):
        img = Image.fromarray(vec)
        img = img.resize((320,320), resample=Image.NEAREST)
        img.save(f'sv_{i+1}.png')

    scaled_w = np.interp(model.w, (model.w.min(), model.w.max()), (0, 255))
    img = Image.fromarray(scaled_w.reshape(32,32,3).astype(np.uint8)).resize((320,320), resample=Image.NEAREST)
    img.save(f'w.png')
