#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys


#########################################
# part (a)
#########################################

def normalize(a):
    return (a-a.mean(axis=0))/a.std()

def phi(Y):
    return np.count_nonzero(Y!=0)/len(Y)

def mu(X,Y):
    X1, X2 = X[Y==0],X[Y!=0]
    return np.array([X1.mean(axis=0),X2.mean(axis=0)])

def S(X,Y):
    X1, X2 = X[Y==0],X[Y!=0]
    m1, m2 = mu(X,Y)
    return ((X1-m1).T@(X1-m1) + (X2-m2).T@(X2-m2))/len(Y)


#########################################
# part (b)
#########################################

def plot_data(X,Y,save_file_name=None):
    X_x, X_y = X.T
    s = plt.scatter(X_x,X_y,c=Y,cmap=ListedColormap(['red','green']))
    
    plt.title('Scatter plot of Datapoints')
    plt.xlabel('x_0 (normalized)')
    plt.ylabel('x_1 (normalized)')
    labels = ['Alaska','Canada']
    plt.legend(handles=s.legend_elements()[0], labels=labels)
    
    if save_file_name:
        plt.savefig(save_file_name, dpi=150, bbox_inches='tight')


#########################################
# part (c)
#########################################

def plot_boundary(X,Y,phi,mu,save_file_name=None):
    X_x, X_y = X.T
    s = plt.scatter(X_x,X_y,c=Y,cmap=ListedColormap(['red','green']))
    
    # getting separator: perp. bisector of mu's
    midpoint = mu[0]*(1-phi) + mu[1]*phi
    diff = (mu[1]-mu[0])
    slope = -diff[1]/diff[0]
    print(midpoint,slope)
    
    plt.title('Scatter plot of Datapoints')
    plt.xlabel('x_0 (normalized)')
    plt.ylabel('x_1 (normalized)')
    plt.axline(midpoint, slope=slope, color='b')
    labels = ['Alaska','Canada']
    plt.legend(handles=s.legend_elements()[0], labels=labels)
    
    if save_file_name:
        plt.savefig(save_file_name, dpi=150, bbox_inches='tight')


#########################################
# part (d)
#########################################

def Si(X,Y):
    X1, X2 = X[Y==0],X[Y!=0]
    m1, m2 = mu(X,Y)
    return (((X1-m1).T@(X1-m1))/len(X1), ((X2-m2).T@(X2-m2))/len(X2))


#########################################
# part (e)
#########################################

def boundary_fn(phi,mu,det_sigma,inv_sigma,X):

    return (X-mu[0]).T@inv_sigma[0]@(X-mu[0]) - (X-mu[1]).T@inv_sigma[1]@(X-mu[1]) \
         + np.log(det_sigma[0]) - np.log(det_sigma[1]) \
         + 2*np.log(phi/(1-phi))

def plot_quadratic_boundary(X,Y,phi,mu,sigma,save_file_name=None):
    X_x, X_y = X.T
    s = plt.scatter(X_x,X_y,c=Y,cmap=ListedColormap(['red','green']))
    
    # getting separator: P(c1) = P(c2). Solve.
    det_sigma = [np.linalg.det(sigma[0]), np.linalg.det(sigma[1])]
    inv_sigma = [np.linalg.inv(sigma[0]), np.linalg.inv(sigma[1])]
    
    x0_space = np.linspace(-0.5,0.5,401)
    x1_space = np.linspace(-1,1,801)
    x_meshgrid = np.stack(np.meshgrid(x0_space, x1_space),axis=2)
    
    vector_boundary_fn = np.vectorize(boundary_fn, signature="(),(2,2),(2),(2,2,2),(2)->()")
    boundary_2d = vector_boundary_fn(phi,mu,det_sigma,inv_sigma,x_meshgrid)
    plot_pts = x_meshgrid[(boundary_2d<0.005)&(boundary_2d>-0.005)].T
    
    plt.plot(plot_pts[0], plot_pts[1], color='b')
    
    plt.title('Scatter plot of Datapoints')
    plt.xlabel('x_0 (normalized)')
    plt.ylabel('x_1 (normalized)')
    labels = ['Alaska','Canada']
    plt.legend(handles=s.legend_elements()[0], labels=labels)
    
    if save_file_name:
        plt.savefig(save_file_name, dpi=150, bbox_inches='tight')


class GaussianDiscriminantAnalyser:
    
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.phi = None
        self.det_sigma = None
        self.inv_sigma = None
        self.bf = np.vectorize(boundary_fn, signature="(),(2,2),(2),(2,2,2),(2)->()")
    
    def fit(self,X,Y):
        self.phi = phi(Y)
        self.mu = mu(X,Y)
        self.sigma = Si(X,Y)
        self.det_sigma = [np.linalg.det(self.sigma[0]), np.linalg.det(self.sigma[1])]
        self.inv_sigma = [np.linalg.inv(self.sigma[0]), np.linalg.inv(self.sigma[1])]
        
    def predict(self,X):
        Y = self.bf(self.phi, self.mu, self.det_sigma, self.inv_sigma, X)
        return np.where(Y>0,'Canada', 'Alaska')


if __name__ == "__main__" and "__file__" in globals():
    if len(sys.argv) < 3:
        print("ERROR: this script requires a train and test directory. Exiting.")
    
    train_dir, test_dir = sys.argv[1],sys.argv[2]
    trainX = normalize(np.loadtxt(f"{train_dir}/X.csv", delimiter=','))
    trainY = np.loadtxt(f"{train_dir}/Y.csv", dtype=object)
    testX = normalize(np.loadtxt(f"{test_dir}/X.csv", delimiter=','))
    
    trainY_enc = np.where(trainY=='Alaska',0,1)

    model = GaussianDiscriminantAnalyser()
    model.fit(trainX,trainY_enc)
    preds = model.predict(testX)
    np.savetxt("result_4.txt",preds,fmt="%s")


