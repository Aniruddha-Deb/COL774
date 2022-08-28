#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


#########################################
# part (a)
#########################################

def sample(n=1000000,theta=np.array([[3],[1],[2]]),x1_par=(3,2),x2_par=(-1,2),sigma=np.sqrt(2)):
    X = np.column_stack([
            np.full(n,1),
            norm.rvs(loc=x1_par[0],scale=x1_par[1],size=n),
            norm.rvs(loc=x2_par[0],scale=x2_par[1],size=n)
        ])
    eps = norm.rvs(loc=0,scale=sigma,size=(n,1))
    
    return (X,X@theta + eps)


#########################################
# part (b)
#########################################

def normalize(a):
    return (a-a.mean())/a.std()

def h(X, theta):
    return X@theta

def J(X, Y, theta):
    return ((Y-X@theta)**2).mean()/2

def grad_J(X, Y, theta):
    return X.T@(X@theta - Y)/len(Y)

def stochastic_gradient_descent(X, Y, theta, batch_size=100, eta=0.01, stop_lim=0.0001, t_lim=100):
    
    theta_pts = [theta]
    
    train_X = X
    train_Y = Y
    
    t = 0
    
    # Convergence: 
    
    while t < t_lim:
                
        for i in range(len(Y)//batch_size):
            X_batch = train_X[i*batch_size:(i+1)*batch_size]
            Y_batch = train_Y[i*batch_size:(i+1)*batch_size]
        
            loss_grad = grad_J(X_batch,Y_batch,theta_pts[-1])
            theta_n = theta_pts[-1] - eta*loss_grad
        
            if (abs(theta_n - theta_pts[-1]).max() < stop_lim):
                theta_pts.append(theta_n)
                return theta_pts

            theta_pts.append(theta_n)

        p = np.random.permutation(len(train_Y))
        train_X = train_X[p]
        train_Y = train_Y[p]
        
        t += 1
        
    if (t == t_lim):
        print("t_lim hit")
            
    return theta_pts


#########################################
# part (d)
#########################################

def plot_paths(paths, elev=45, azim=60, save_file_name=None):
    
    fig = plt.figure(figsize=(5,5), dpi=150)
    ax = fig.add_axes([0,0,1,1], projection='3d')
    ax.view_init(elev=elev, azim=azim)
    #surf = ax.plot_surface(t0_space, t1_space, loss_space, cmap='viridis', edgecolor=None, alpha=alpha)
    #surf.set_facecolor((0,0,0,0))
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    ax.set_zlabel('theta_2')
    
    for path in paths:
        path = path[::20]
        desc_x, desc_y, desc_z = np.stack(path,axis=0).reshape((len(path),3)).T
        ax.plot(desc_x, desc_y, desc_z, marker='.', alpha=0.5)


