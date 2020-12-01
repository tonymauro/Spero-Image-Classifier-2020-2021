import numpy as np
from tqdm import tqdm
import sys
import math

from sklearn.datasets.samples_generator import make_blobs

class DeD_Enumerator():
    def __init__(self, data):
        # data is a numpy array
        self.data = data
        self.depths = self.mahalanobis(data)

    def mahalanobis(self, data):
        """
        calculates the Mahalanobis depth for each point in the dataset
        """
        covariance_mat = np.cov(data.T)
        try:
            inv_covmat = np.linalg.inv(covariance_mat)
        except np.linalg.LinAlgError:
            print("Could not invert covariance matrix")
            sys.exit(0)
        
        """returns an n_sample by n_features array
        # each feature in each sample is subtracted from the average value of that feature
        across all samples"""
        x_minus_mu = data - np.mean(data, axis=0)
        # if the array is one dimensional make it 2d so it can be transposed
        mult = np.matmul(x_minus_mu, inv_covmat)
        if x_minus_mu.shape[0] == 1:
            x_minus_mu = [x_minus_mu]
        mult = np.matmul(mult, x_minus_mu.T)

        try:
            mahal = np.linalg.inv(mult + 1)
        except np.linalg.LinAlgError:
            print("Could not invert final matrix")
            sys.exit(0)

        return mahal.diagonal()

    def optimal_k(self, krange):
        """
        returns the optimal number of clusters, k
        """
        depths = self.depths
        depth_median = self.depth_median(depths)
        avg_delta = self.avg_delta(depths, depth_median)
        depth_diffs = []
        # calculating depth difference for different number of clusters
        for k in tqdm(krange):
            rng = math.floor(self.data.shape[0]/k)
            start = 0
            end = 0
            cluster_depth_medians = []
            cluster_avg_deltas = []
            # going through each of the k clusters
            for j in range(1, k+1):
                # paritioning data into k group; these are the clusters
                start = end
                end = start + rng
                if j == k:
                    end += 1
                cluster_depths = depths[start:end]
                cluster_depth_median = self.depth_median(cluster_depths)
                cluster_avg_delta = self.avg_delta(cluster_depths, depth_median)
                cluster_depth_medians.append(cluster_depth_median)
                cluster_avg_deltas.append(cluster_avg_delta)
            depth_within = self.depth_within(cluster_avg_deltas)
            depth_between = self.depth_between(avg_delta, depth_within)
            depth_diff = self.depth_diff(depth_within, depth_between)
            # adding depth diff for this k value to list of all depth diffs
            depth_diffs.append(depth_diff)
        # finds index of maximum value in list
        optimal_k_index = depth_diffs.index(max(depth_diffs))
        return krange[optimal_k_index]

    def depth_diff(self, depth_within, depth_between):
        """
        returns difference between depth_within and depth_between
        """
        return depth_within - depth_between

    def depth_between(self, avg_delta, depth_within):
        """
        returns difference between avg_delta and depth_within
        """
        return avg_delta - depth_within

    def depth_within(self, avg_deltas):
        """
        returns average of all values in avg_deltas list
        """
        return np.average(avg_deltas)
    
    def avg_delta(self, depths, median):
        """
        finds average difference between each depth value and the depth median
        """
        diffs = np.abs(np.array(depths) - median)
        return np.average(diffs)
    
    def depth_median(self, depths):
        """
        returns the point with the maximum depth in the dataset
        """
        return np.max(depths)

dataset = make_blobs(n_samples=100, n_features=5, centers=4)
ded = DeD_Enumerator(dataset[0])
#print(dataset[0])
optimalk = ded.optimal_k(range(2, 11))
print(optimalk)
