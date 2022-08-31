#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


#########################################
# part (a)
#########################################

def normalize(a):
    return (a-a.mean())/a.std()

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


