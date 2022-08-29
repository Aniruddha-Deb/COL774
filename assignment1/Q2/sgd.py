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

def stochastic_gradient_descent(X, Y, theta, batch_size=100, eta=0.001, stop_lim=0.000001, t_lim=100):
    
    theta_pts = [theta]
    losses = []
    
    train_X = X
    train_Y = Y
    
    t = 0
    
    # Convergence: 
    # Unlike gradient descent, for SGD, converge when the ratio of the change is small enough. Compute after
    # going over (atleast) 1000 samples.
    
    # this prevent the algorithm from taking too much time or if it's stuck without converging
    print(f"len_y = {len(Y)}")
    print(len(Y)//batch_size)
    while t < t_lim:
        
        n_samples = 0
        n_iters = 0
        loss_av = 0
        prev_loss_av = None
        
        for i in range(len(Y)//batch_size):
            X_batch = train_X[i*batch_size:(i+1)*batch_size]
            Y_batch = train_Y[i*batch_size:(i+1)*batch_size]
        
            loss_grad = grad_J(X_batch,Y_batch,theta_pts[-1])
            theta_n = theta_pts[-1] - eta*loss_grad
            n_samples += batch_size
            n_iters += 1
            loss_av += J(X_batch,Y_batch,theta_pts[-1])
            
            if (n_samples >= 1000):
                loss_av /= n_iters
                if prev_loss_av:
                    # compute ratio of loss of previous theta and current theta
                    # if it's less than the stop limit, then stop.
                    if (loss_av < prev_loss_av and abs(loss_av-prev_loss_av) < stop_lim) and loss_av < 1):
                        theta_pts.append(theta_n)
                        return (theta_pts,losses)
                    
                losses.append(loss_av)
                prev_loss_av = loss_av
                loss_av = 0
                n_samples = 0
                n_iters = 0
        
            theta_pts.append(theta_n)

        # random shuffle for the next epoch
        p = np.random.permutation(len(train_Y))
        train_X = train_X[p]
        train_Y = train_Y[p]
        
        t += 1
        
    if (t == t_lim):
        print("t_lim hit")
    
    return (theta_pts,losses)


#########################################
# part (d)
#########################################

def plot_paths(paths, elev=45, azim=60, save_file_name=None):
    
    fig = plt.figure(figsize=(5,5), dpi=150)
    ax = fig.add_axes([0,0,1,1], projection='3d')
    ax.view_init(elev=elev, azim=azim)
    #surf = ax.plot_surface(t0_space, t1_space, loss_space, cmap='viridis', edgecolor=None, alpha=alpha)
    #surf.set_facecolor((0,0,0,0))
    ax.set_title('Paths taken by gradient descent for varying values of b')
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    ax.set_zlabel('theta_2')
    
    for (i,path) in enumerate(paths):
        path = path[::20]
        desc_x, desc_y, desc_z = np.stack(path,axis=0).reshape((len(path),3)).T
        ax.plot(desc_x, desc_y, desc_z, marker='.', alpha=0.5, label=f'b={10**(i*2)}')
        
    if save_file_name:
        fig.savefig(save_file_name, bbox_inches='tight')
    
    ax.legend()


