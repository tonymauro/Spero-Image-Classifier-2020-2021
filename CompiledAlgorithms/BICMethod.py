
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

class BIC:
    def __init__(self, cluster_range):
        self.cluster_range = cluster_range
    def bicMethod(self, data):
        bics = []
        for n_clusters in tqdm(self.cluster_range):
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
            gmm.fit(data)
            bics.append(gmm.bic(data))

        print(bics)
        return self.cluster_range[bics.index(min(bics))]
