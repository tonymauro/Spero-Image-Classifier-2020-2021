import matplotlib.pyplot as plt
import pandas
import csv
import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from tqdm import tqdm

from .elbowMethod import Elbow
from .ENVI_Files import Envi
from .ClusteringAlgorithm import ClusteringAlgorithm

class DominantColorsGMM(ClusteringAlgorithm):

    def __init__(self, path, imageName, resultFolderDir, cluster_override, decimate_factor, cluster_enum='elbow'):
        super().__init__(path, imageName, resultFolderDir, cluster_override, decimate_factor, cluster_enum=cluster_enum)
        self.ALG = 'GMM'

    def cluster(self, img):
        # Runs the Scikitlearn algorithm with the determined number of clusters
        gmm = GaussianMixture(n_components=self.CLUSTERS, n_init=3, covariance_type='full', verbose=1)
        gmm.fit(img)

        # Centroids are the "average clusters" of each cluster
        # Labels are numbers denoting which cluster each pixel belongs to (Pixel location corresponds with the label's
        # index.
        # Ex. Label = [0, 4, 1, 2, 3, 1, 4, 0, 0..... 1, 2], where 0 would be cluster 0, 1 would be cluster 1, etc...
        # and pixel at (0, 0) would belong to cluster 0.
        self.LABELS = gmm.predict(img)
        self.CENTROIDS = self.find_centers()

    def findDominant(self):
        img = super().findDominant()
        try:
            self.cluster(img)
        except NameError:
            print("not found")
        self.plot()
        return self.RESULT_PATH