from CompiledAlgorithms.HierarchicalAlgorithm import HierarchicalClusterAlgorithm
from CompiledAlgorithms.KMeansAlgorithm import DominantColorsKMeans
from CompiledAlgorithms.GMMAlgorithm import DominantColorsGMM
from CompiledAlgorithms.CustomKMeansAlgorithm import KMeans
from CompiledAlgorithms.elbowMethod import Elbow


class AlgorithmRunner:
    # Runs selected algorithm with optional overrides
    # Hierarchical has n-neighbors, but not KMeans
    def run_HierarchicalClusterAlgorithm(self, image_path, image_name, result_folder, cluster_override=0,
                                         n_neighbors_override=0, decimate_factor=1, no_results_output=False):
        alg = HierarchicalClusterAlgorithm(image_path, image_name, result_folder, cluster_override,
                                     n_neighbors_override, decimate_factor, no_results_output)


    def run_kMeansAlgorithm(self, image_path, image_name, result_folder, cluster_override=0, decimate_factor=1):
        algorithm = DominantColorsKMeans(image_path, image_name, result_folder, cluster_override, decimate_factor)
        return algorithm.findDominant()
    def run_GMMAlgorithm(self, image_path, image_name, result_folder, cluster_override=0, decimate_factor=1):
        algorithm = DominantColorsGMM(image_path, image_name, result_folder, cluster_override, decimate_factor)
        return algorithm.findDominant()

    def run_CustomKMeansAlgorithm(self, image_path, filename , result_folder, custom_clusters=0, decimation=1, max_iterations=30):
        km = KMeans()
        print(image_path)
        path = image_path
        if custom_clusters == 0:
            elbowM = Elbow()
            km.k = km.elbow(10, 30, path, decimation, result_folder, filename)

        else:
            km.k = custom_clusters
        km.initialize(path, decimation, result_folder, filename)
        km.train(max_iterations)
        km.outputImageAlternate()
        km.plot()
        km.makeCSV()
