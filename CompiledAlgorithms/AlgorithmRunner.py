from CompiledAlgorithms.HierarchicalAlgorithm import HierarchicalClusterAlgorithm
from CompiledAlgorithms.KMeansAlgorithm import DominantColorsKMeans
from CompiledAlgorithms.GMMAlgorithm import DominantColorsGMM
from CompiledAlgorithms.EXPAlgorithm import DominantColorsEXP
from CompiledAlgorithms.CustomKMeansAlgorithm import KMeans
from CompiledAlgorithms.elbowMethod import Elbow


class AlgorithmRunner:
    # Runs selected algorithm with optional override

    def runSklearnAlg(self, alg, path, imageName, resultFolderDir, cluster_enum, norm, cluster_override=0, decimate_factor=1):
        """
        Method to run the sklearn clustering algorithm specified in the GUI
        """
        # adding the parameters to a dictionary that can be passed to each algorithm class
        kwargs = {
            "path": path,
            "imageName": imageName,
            "resultFolderDir": resultFolderDir,
            "cluster_override": cluster_override,
            "decimate_factor":decimate_factor, 
            "cluster_enum": cluster_enum,
            "norm": norm,
            "alg": 'K-Means'
        }
        # using the selected algorithm
        if alg == 'kmeans':
            algorithm = DominantColorsKMeans(**kwargs)
        elif alg == 'gmm':
            algorithm = DominantColorsGMM(**kwargs)
        elif alg == 'exp':
            algorithm = DominantColorsEXP(**kwargs)
        
        return algorithm.findDominant()

    # Hierarchical has n-neighbors, but not KMeans
    def run_HierarchicalClusterAlgorithm(self, image_path, image_name, result_folder, cluster_override=0,
                                         n_neighbors_override=0, decimate_factor=1, no_results_output=False):
        alg = HierarchicalClusterAlgorithm(image_path, image_name, result_folder, cluster_override,
                                     n_neighbors_override, decimate_factor, no_results_output)

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
