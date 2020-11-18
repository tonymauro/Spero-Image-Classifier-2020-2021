import time
import hashlib
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import pairwise_distances


class Gap:

    def __init__(self):
        return

    # Method to compute the inertia for the given cluster and dataset
    def getInertia(self, a, X):
        kmeanmodel = KMeans(n_clusters=np.max(a)+1).fit(X)
        return kmeanmodel.inertia_
    """
    def getInertia(self, a, X):
        z = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
        return np.mean(z)
    """
    # Method to compute the gap statistic of the given (data), with the given clustering (algorithm) ie: KMeans()
    # (kmax) represents the upper limit of the number of clusters
    # (nrefs) represents the number of reference data sets that will be created
    def getGap(self, algorithm, data, kmax=11, nrefs=5):

        # conforms extreme inputs
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # generates reference data set
        ref = np.random.rand(*data.shape)

        # computes inertia of reference data set
        ref_inertia = []
        for k in range(1, kmax + 1):
            inertia = []
            for _ in range(nrefs):
                algorithm.n_clusters = k
                assignments = algorithm.fit_predict(ref)
                inertia.append(self.getInertia(assignments, ref))
            ref_inertia.append(np.mean(inertia))

        # computes inertia of given data set
        data_inertia = []
        for k in range(1, kmax):
            algorithm.n_clusters = k
            assignments = algorithm.fit_predict(data)
            data_inertia.append(self.getInertia(assignments, data))

        # creates an array of gap values for all k, using the gap statistics given formula
        gaps = np.log(np.mean(ref_inertia)) - np.log(data_inertia)

        # returns the index of the largest gap value, representing the determined optimal number of clusters
        return gaps.argmax() + 1