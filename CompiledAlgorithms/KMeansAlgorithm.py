import matplotlib.pyplot as plt
import pandas
import csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import datetime

from .ClusteringAlgorithm import ClusteringAlgorithm
from .ENVI_Files import Envi
from .elbowMethod import Elbow


class DominantColorsKMeans(ClusteringAlgorithm):
    def __init__(self, path, imageName, resultFolderDir, cluster_override, decimate_factor):
        super().__init__(path, imageName, resultFolderDir, cluster_override, decimate_factor)
        self.ALG = 'K-Means'

    def find_centers(self):
        """
        finds average of spectra of all pixels in a cluster
        """
        points = self.origImage
        c = 0
        centers = np.zeros((self.CLUSTERS, points.shape[2]))
        counts = np.zeros(self.CLUSTERS)
        for x in range(points.shape[0]):
            for y in range(points.shape[1]):
                centers[self.LABELS[c]] += points[x,y]
                counts[self.LABELS[c]] += 1
                c += 1
        for i in range(self.CLUSTERS):
            centers[i] /= counts[i]
        print(centers)
        return centers

    def cluster(self, img):
        # Runs the Scikitlearn algorithm with the determined number of clusters
        kmeans = KMeans(n_clusters=self.CLUSTERS, n_init=20)
        kmeans.fit(img)

        # Centroids are the "average clusters" of each cluster
        # Labels are numbers denoting which cluster each pixel belongs to (Pixel location corresponds with the label's
        # index.
        self.LABELS = kmeans.labels_
        self.CENTROIDS = self.find_centers()

    def findDominant(self):
        img = super().findDominant()
        try:
            self.cluster(img)
        except NameError:
            print("not found")
        self.plot()
        return self.RESULT_PATH