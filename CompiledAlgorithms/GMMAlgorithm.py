import matplotlib.pyplot as plt
import pandas
import csv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from CompiledAlgorithms.elbowMethod import Elbow
import numpy as np
from sklearn.decomposition import PCA
from CompiledAlgorithms.ENVI_Files import Envi
import datetime

class DominantColors1:
    CLUSTERS = None
    IMAGE = None
    CENTROIDS = None
    LABELS = None
    WAVELENGTHS = None
    PCAON = True
    PCADIMENSIONS = 0
    def __init__(self, path, imageName, resultFolderDir, cluster_override, decimate_factor):
        print(datetime.datetime.now())
        # PATH is the path of the ENVI File, RESULT_PATH is where the results will be saved
        # Cluster overrides and decimation factor are optional override options for user
        # Default resultFolderDir = ...CompiledAlgorithms/Result/imageName+timestamp
        self.IMAGE = None
        self.imageName = imageName
        self.origImage = path

        self.PATH = path
        self.RESULT_PATH = resultFolderDir

        self.cluster_override = cluster_override
        self.decimate_factor = decimate_factor

    def find_centers(self):
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

    def findDominant(self):
        '''
        Uses elbow method to find number of clusters and then applys GMM
        '''
        # Creates ENVI Object and reads the image at the given path
        ei = Envi.EnviImage()
        ei.Read(self.PATH, True, False, False)

        # Decimates spectrum based on optional input(Default is 1, does not affect data)
        ei.DecimateSpectrum(self.decimate_factor)

        # Saves original image for later use in display
        self.origImage = ei.Pixels
        # Supposed to be wavenumbers but ENVI reader's variable is called wavelengths
        self.WAVELENGTHS = ei.wavelength
        # reshapes image into 2D array shape (x*y, z)
        # Kmeans only takes in 2D arrays
        img = np.array(ei.Pixels)
        img = img.reshape((ei.Pixels.shape[0] * ei.Pixels.shape[1], ei.Pixels.shape[2]))
        origimage = img
        if self.PCAON:
            pca = PCA()
            pca.fit(img)
            # print(pca.explained_variance_ratio_)
            sum = 0
            for x in range(len(pca.explained_variance_ratio_)):
                if pca.explained_variance_ratio_[x] > 0.001:
                    sum += pca.explained_variance_ratio_[x]
                    self.PCADIMENSIONS += 1
                else:
                    break
            print(sum)
            print(self.PCADIMENSIONS)
            pca = PCA(n_components=self.PCADIMENSIONS)
            pca.fit(img)
            print(pca.explained_variance_ratio_)
            img = pca.fit_transform(img)
        self.IMAGE = img

        # Kmeans clustering
        # Uses elbow method to calculate the optimal K clusters (Unless override by user, where cluster_override != 0)
        if self.cluster_override == 0:
            self.CLUSTERS = Elbow.elbowMethod(self, self.IMAGE)
        else:
            self.CLUSTERS = self.cluster_override

        # Runs the Scikitlearn algorithm with the determined number of clusters
        gmm = GaussianMixture(n_components=self.CLUSTERS, n_init=1, covariance_type='full')
        gmm.fit(img)



        # Centroids are the "average clusters" of each cluster
        # Labels are numbers denoting which cluster each pixel belongs to (Pixel location corresponds with the label's
        # index.
        # Ex. Label = [0, 4, 1, 2, 3, 1, 4, 0, 0..... 1, 2], where 0 would be cluster 0, 1 would be cluster 1, etc...
        # and pixel at (0, 0) would belong to cluster 0.
        self.LABELS = gmm.predict(img)
        self.CENTROIDS = self.find_centers()
        # Creates color coded image based on the clusters and a
        # centroid graph plotting each cluster's average spectrum
        self.plot()

        return self.RESULT_PATH

    def plot(self):
        points = self.origImage
        centroids = self.CENTROIDS
        k = self.CLUSTERS
        labels = self.LABELS
        # Temporary solution for color key. Require better method of creating differentiable colors
        # (Currently only selects colors from the colorChoices array)
        # If number of clusters > len(colorChoices) algorithm output would be wrong.
        colorChoices = [[0, 1, 0], [1, 0, 0], [0, 1, 1], [0.5, 0.5, 0], [1, 0, 1], [1, .5, 1], [1, 0, 1], [0, 0, 1],
                        [.5, .5, .5], [.5, .5, 1]]
        # Plots the wavenumber vs absorption graph for each centroid color coded
        plt.figure()
        plt.ylabel('Absorption')
        plt.xlabel('Wave Number')
        plt.title("Wave Number vs Absorption For Each Center(KMeans)")
        for center in range(k):
            plt.plot(self.WAVELENGTHS, self.CENTROIDS[center], color=colorChoices[center], marker="o",
                     label="Center " + str(center + 1))
        plt.legend()
        plt.savefig(self.RESULT_PATH + self.imageName + "_ClusteredAbsorptionGraph.png")

        newImg = np.zeros((points.shape[0], points.shape[1], 3))
        # Remakes the image based on the labels at each pixel
        # The label's color is determined by the index of list colorChoices
        c = 0
        for x in range(newImg.shape[0]):
            for y in range(newImg.shape[1]):
                newImg[x, y] = colorChoices[labels[c]]
                c += 1
        # Plots the 3D graph using R G B list collected from above and use colors from the clusters list
        # Saves the image as a png file in the result folder given from user input(If no user input, the
        # files would be saved at default "RESULT/*FILENAME*"
        # Creates image without borders or axis
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(newImg)
        plt.savefig(self.RESULT_PATH + self.imageName + "_ClusteredImage.png", bbox_inches="tight", pad_inches=0)

        self.makeCSV()
        print(datetime.datetime.now())

    # no change for GMM but we will change later
    def makeCSV(self):
        # makes CSV file of the clustered data in the given directory in case
        # user wants to further analyze the chemical composition
        path = self.RESULT_PATH + self.imageName + '.csv'

        with open(path, mode='w+', newline='') as csvFile:
            fieldnames = ['Wavelength']
            for x in range(len(self.CENTROIDS)):
                fieldnames.append("Cluster " + str(x + 1))

            writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            writer.writeheader()
            for wl in self.WAVELENGTHS:
                tempDict = {"Wavelength": wl}
                for x in range(len(self.CENTROIDS)):
                    tempDict["Cluster " + str(x + 1)] = self.CENTROIDS[x][self.WAVELENGTHS.index(wl)]
                writer.writerow(tempDict)

        df = pandas.read_csv(path)
        print(df)
