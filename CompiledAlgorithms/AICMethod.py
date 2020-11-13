
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

class AIC:
    def __init__(self, cluster_range):
        self.cluster_range = cluster_range
    def aicMethod(self, data):
        aics = []
        for n_clusters in tqdm(self.cluster_range):
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
            gmm.fit(data)
            aics.append(gmm.aic(data))


        print(aics)
        return self.cluster_range[aics.index(min(aics))]
