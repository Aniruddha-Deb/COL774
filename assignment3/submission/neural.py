import numpy as np
import pandas as pd
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time
import os

from contextlib import redirect_stdout

def sigmoid(x):
    return np.tanh(x*0.5)*0.5 + 0.5 # numerically stable

def sigmoid_dv(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def relu_dv(x):
    return np.heaviside(x,0.5)

def leaky_relu(x):
    return np.maximum(0,x)+np.minimum(x,0)*(0.1)

def leaky_relu_dv(x):
    return np.heaviside(x,0.05)+(1-np.heaviside(x,0.05))*0.1

def MSE_loss(Y_true, Y_pred):
    return ((Y_pred-Y_true)**2).sum(axis=1).mean()/2

def MSE_loss_dv(Y_true, Y_pred):
    return (Y_pred-Y_true)

def CE_loss(y_true, y_pred):
    return -(y_true*np.log(y_pred+1e-15) + (1-y_true)*np.log(1-y_pred+1e-15)).sum(axis=1).mean()

def CE_loss_dv(y_true, y_pred):
    return -y_true/(y_pred+1e-15) + (1-y_true)/(1-y_pred+1e-15)

def constant_lr(epoch, batch):
    return 0.1

def epoch_lr(epoch, batch):
    return 0.1/np.sqrt(epoch+1)

def print_mat_list(name, l):
    print(f"{name}: [{', '.join([str(w.shape) for w in l])}]")

class NeuralNetwork:
    
    def __init__(self, M, n, layers, r, activation=sigmoid, activation_dv=sigmoid_dv, loss=MSE_loss, loss_dv=MSE_loss_dv, lr_fn=constant_lr):
        l = [n] + layers + [r]
        self.weights = [np.random.randn(l[i], l[i+1])*np.sqrt(2/l[i+1]) for i in range(len(l)-1)]
        print_mat_list("wt", self.weights)
        self.biases = [np.random.randn(l[i+1])*np.sqrt(2/l[i+1]) for i in range(len(l)-1)]
        print_mat_list("bias", self.biases)
        self.activations = [activation]*len(layers) + [sigmoid]
        print(self.activations)
        self.activation_dvs = [activation_dv]*len(layers) + [sigmoid_dv]
        print(self.activation_dvs)
        self.Z = []
        self.L = []
        self.loss = loss
        self.loss_dv = loss_dv
        self.n = len(self.weights)
        self.lr_fn = lr_fn
        self.M = M
    
    def backward(self, y):
        # returns weight matrix derivatives with the shapes (n,l_1), (l_1,l_2), \ldots, (l_L,r) and
        # bias derivatives with the shapes l_1, l_2, \ldots, l_L, r
        
        wt_dv = []
        bias_dv = []
        
        dL = self.loss_dv(y,self.L[-1])
        for i in range(self.n-1,-1,-1):
            dZ = self.activation_dvs[i](self.Z[i])
            wt_dv.append(self.L[i].T@(dL*dZ))
            bias_dv.append(dL*dZ)
            dL = (dL*dZ)@self.weights[i].T
        
        return wt_dv[::-1], bias_dv[::-1]

    def forward(self, X):
        # populates self.Z and self.L with outputs of each layer, and finally returns the output of the 
        # final layer
        
        self.L = []
        self.Z = []
        self.L.append(X)
        for i in range(self.n):
            self.Z.append(self.L[-1]@self.weights[i] + self.biases[i])
            self.L.append(self.activations[i](self.Z[-1]))
            
        return self.L[-1]

    def backpropagate(self, wt_dv, bias_dv, eta=0.1):
        for (i,(dw,db)) in enumerate(zip(wt_dv,bias_dv)):
            self.weights[i] -= eta*(dw/self.M)
            self.biases[i] -= eta*(db.mean(axis=0))
        
    def fit(self, X, y):
        
        y_dec = np.argmax(y,axis=1).flatten()
        l = None
        size = X.shape[0]
        prev_loss_diff = 1e9
        
        for epoch in range(5):
            X_t, y_t = shuffle(X,y)
            train_loss = 0
            n_batches = X_t.shape[0]//self.M
            for batch in range(n_batches):
                preds = self.forward(X_t[batch*self.M:(batch+1)*self.M])
                wt_dv, bias_dv = self.backward(y_t[batch*self.M:(batch+1)*self.M])
                self.backpropagate(wt_dv, bias_dv, eta=self.lr_fn(epoch, batch))
                train_loss += self.loss(y_t[batch*self.M:(batch+1)*self.M], preds)

            train_loss /= n_batches
            
            if (epoch % 10 == 9):
                print(f"Epoch {epoch+1}")
                print(f"  Train Loss: {train_loss}")
                print("")
                
            if epoch > 50 and np.abs(train_loss/l - 1) < 1e-4:
                print(f"Termination condition reached at epoch {epoch+1}. Exiting.")
                break
            l = train_loss
        
    def predict(self, X):
        return np.argmax(self.forward(X),axis=1).flatten()

def test_model_params(X_train, y_train, y_train_onehot, X_test, y_test, outpath, name, metadata, dryrun=False):
    if not dryrun:
        os.makedirs(f"{outpath}/{name}", exist_ok=True)
    path = f"{outpath}/{name}"
    train_accs = {}
    test_accs = {}
    train_times = {}
    for label in metadata:
        data = metadata[label]
        arglabels = ['activation', 'activation_dv', 'loss', 'loss_dv', 'lr_fn']
        net = NeuralNetwork(100, 784, data['layers'], 10, **{k:data[k] for k in arglabels})
        tic = time.perf_counter()
        net.fit(X_train, y_train_onehot)
        toc = time.perf_counter()
        train_acc = accuracy_score(y_train,net.predict(X_train))
        test_acc = accuracy_score(y_test,net.predict(X_test))
        
        cmat = ConfusionMatrixDisplay.from_predictions(net.predict(X_test), y_test)
        if not dryrun:
            cmat.figure_.savefig(f"{path}/cmat_{label}.pdf", bbox_inches='tight')
        train_accs[label] = train_acc
        test_accs[label] = test_acc
        train_times[label] = toc-tic

    print(f"TrainAcc TestAcc Time")
    for label in metadata:
        print(f"{train_accs[label]:.4f} {test_accs[label]:.4f} {train_times[label]:.2f}s")
    
    fig, ax = plt.subplots(1,2,dpi=150,figsize=(12,4))
    ax[0].plot(*zip(*train_accs.items()), label='Train', marker='.')
    ax[0].set_title("Accuracy vs NN Layout")
    ax[0].set_xlabel("Layout")
    ax[0].set_ylabel("Accuracy")
    ax[0].plot(*zip(*test_accs.items()), label='Test', marker='.')
    ax[0].legend()
    ax[1].set_title("Training Time vs NN Layout")
    ax[1].set_ylabel("Training time (s)")
    ax[1].set_xlabel("Layout")
    ax[1].plot(*zip(*train_times.items()), marker='.')
    if not dryrun:
        fig.savefig(f'{path}/acc_time.pdf', bbox_inches='tight')
    
    best_net = max(test_accs, key=test_accs.get)
    print(f"The best network is {best_net}")

# # Testing

def load_dataset(path):
    mat = np.loadtxt(path, delimiter=",")
    X,y = mat[:,:28*28],mat[:,28*28].flatten().astype(np.int32)
    y_enc = np.eye(10)[y]
    return X/255,y,y_enc

metadata = {
    'b': {
        '5': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [5]
        },
        '10': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [10]
        },
        '15': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [15]
        },
        '20': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [20]
        },
        '25': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [25]
        }
    },
    'c': {
        '5': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : epoch_lr,
            'layers' : [5]
        },
        '10': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : epoch_lr,
            'layers' : [10]
        },
        '15': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : epoch_lr,
            'layers' : [15]
        },
        '20': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : epoch_lr,
            'layers' : [20]
        },
        '25': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : epoch_lr,
            'layers' : [25]
        }
    },
    'd': {
        'relu': {
            'activation' : relu,
            'activation_dv' : relu_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : epoch_lr,
            'layers' : [100,100]
        },
        'sigmoid': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : epoch_lr,
            'layers' : [100,100]
        }
    },
    'e': {
        '2': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [50,50]
        },
        '3': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [50,50,50]
        },
        '4': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [50,50,50,50]
        },
        '5': {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [50,50,50,50,50]
        }
    },
    'f': {
        'best_ce': {
            'activation' : relu,
            'activation_dv' : relu_dv,
            'loss' : CE_loss,
            'loss_dv' : CE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [100]
        }
    }
}


if __name__ == "__main__":

    X_train, y_train, y_train_onehot = load_dataset(sys.argv[1])
    X_test, y_test, y_test_onehot = load_dataset(sys.argv[2])
    outpath = sys.argv[3]
    part = sys.argv[4]

    params = {
            'activation' : sigmoid,
            'activation_dv' : sigmoid_dv,
            'loss' : MSE_loss,
            'loss_dv' : MSE_loss_dv,
            'lr_fn' : constant_lr,
            'layers' : [500,50]
        }
    test_model_params(X_train, y_train, y_train_onehot, X_test, y_test, outpath, "eval", { 'test': params })

#    if part == 'a':
#        pass
#    elif part == 'g':
#        with open(f"{outpath}/{part}.txt", "w") as logfile:
#            with redirect_stdout(logfile):
#                tic = time.perf_counter()
#                clf = MLPClassifier(solver='sgd').fit(X_train, y_train)
#                toc = time.perf_counter()
#                print("{toc-tic:.2f}")
#                clf.score(X_test, y_test)
#                clf.score(X_train, y_train)
#                print(f"{toc-tic:.2f}")
#    else:
#        with open(f"{outpath}/{part}.txt", "w") as logfile:
#            with redirect_stdout(logfile):
#                test_model_params(X_train, y_train, y_train_onehot, X_test, y_test, outpath, part, metadata[part])
#
