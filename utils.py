from __future__ import division
import os
import zipfile
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import scipy.io
import torch
from torch import nn


"""
Geographical information calculation
"""
def get_long_lat(sensor_index,loc = None):
    """
        Input the index out from 0-206 to access the longitude and latitude of the nodes
    """
    if loc is None:
        locations = pd.read_csv('data/metr/graph_sensor_locations.csv')
    else:
        locations = loc
    lng = locations['longitude'].loc[sensor_index]
    lat = locations['latitude'].loc[sensor_index]
    return lng.to_numpy(),lat.to_numpy()

def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000


"""
Load datasets
"""

def load_metr_la_rdata():
    if (not os.path.isfile("data/metr/adj_mat.npy")
            or not os.path.isfile("data/metr/node_values.npy")):
        with zipfile.ZipFile("data/metr/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/metr/")

    A = np.load("data/metr/adj_mat.npy")
    X = np.load("data/metr/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    return A, X

def load_pems_rdata(dataset):

    A = np.load("data/"+dataset+"/adj_mat.npy")
    X = np.load("data/"+dataset+"/node_values.npy").transpose((1,0))
    X = X.astype(np.float32)

    return A, X


def load_sedata():
    assert os.path.isfile('data/sedata/A.mat')
    assert os.path.isfile('data/sedata/mat.csv')
    A_mat = scipy.io.loadmat('data/sedata/A.mat')
    A = A_mat['A']
    X = pd.read_csv('data/sedata/mat.csv',index_col=0)
    X = X.to_numpy()
    return A,X


"""
Dynamically construct the adjacent matrix
"""

def get_Laplace(A):
    """
    Returns the laplacian adjacency matrix. This is for C_GCN
    """
    if A[0, 0] == 1:
        A = A - np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


