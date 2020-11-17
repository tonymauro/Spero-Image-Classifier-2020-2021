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

    def getInertia(self,a,X):
        z = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
        return np.mean(z)

    def getGap(self, algorithm, data, kmax=11, nrefs=5):

        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        ref = np.random.rand(*data.shape)
        refInertia = []
        for k in range (1, kmax+1):
            inertia = []
            for _ in range (nrefs):
                algorithm.n_clusters = k
                assignments = algorithm.fit_predict(ref)
                inertia.append(self.getInertia(assignments, ref))
            refInertia.append(np.mean(inertia))

        dataInertia = []
        for k in range(1, kmax):
            algorithm.n_clusters = k
            assignments = algorithm.fit_redict(data)
            dataInertia.append(self.getInertia(assignments, data))

        gaps = np.log(np.mean(refInertia)) - np.log(dataInertia)

        return (gaps.argmax() + 1)