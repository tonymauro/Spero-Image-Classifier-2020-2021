import matplotlib.pyplot as plt
import pandas
import csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

from .ENVI_Files import Envi
from .elbowMethod import Elbow
from .silhouetteMethod import Silhouette
from .BICMethod import BIC
from .AICMethod import AIC
from .GapStatisticMethod import Gap
from .DeD_enumeration import DeD_Enumerator
import datetime


class ClusteringAlgorithm:

    def __init__(self, path, imageName, resultFolderDir, cluster_override, decimate_factor, PCAON, cluster_enum='elbow', norm='off', alg=None):
        print(datetime.datetime.now())
        # PATH is the path of the ENVI File, RESULT_PATH is where the results will be saved
        # Cluster overrides and decimation factor are optional override options for user
        # Default resultFolderDir = ...CompiledAlgorithms/Result/imageName+timestamp
        self.IMAGE = None
        self.imageName = imageName
        self.origImage = path
        self.normalizedImage = None

        # image info
        self.CLUSTERS = None
        self.CENTROIDS = None
        self.LABELS = None
        self.WAVELENGTHS = None

        self.PATH = path
        self.RESULT_PATH = resultFolderDir

        self.cluster_override = cluster_override
        self.decimate_factor = decimate_factor

        # PCA configs
        self.PCAON = PCAON
        self.PCADIMENSIONS = 0

        #normalize
        self.NORMALIZE = norm

        # cluster enum
        self.CLUSTER_ENUM = cluster_enum

        self.ALG = alg

    def find_centers(self):
        """
        finds average of spectra of all pixels in a cluster
        """
        points = self.origImage
        if self.NORMALIZE != 'off':
            points = self.normalizedImage
        c = 0
        self.CLUSTERS = np.max(self.LABELS) + 1
        centers = np.zeros((self.CLUSTERS, points.shape[2]))
        counts = np.zeros(self.CLUSTERS)
        # averaging the absorbance values of the pixels in each cluster
        for x in range(points.shape[0]):
            for y in range(points.shape[1]):
                centers[self.LABELS[c]] += points[x, y]
                counts[self.LABELS[c]] += 1
                c += 1
        for i in range(self.CLUSTERS):
            centers[i] /= counts[i]
        # print(centers)
        return centers

    def normalize(self):
        img = np.array(self.origImage)

        if self.NORMALIZE == 'mean':
            print("Normalizing Pixels with StandardScaler")
            scaler = StandardScaler()
            for i in tqdm(range(img.shape[0])):
                for j in range(img.shape[1]):
                    img[i, j] = np.transpose(scaler.fit_transform(np.transpose([img[i, j]])))[0]
        elif self.NORMALIZE == 'linear':
            print("Normalizing Pixels with Linear regression")
            x = np.transpose([self.WAVELENGTHS])
            scaler = StandardScaler()
            for i in tqdm(range(img.shape[0])):
                for j in range(img.shape[1]):
                    model = LinearRegression().fit(x, img[i, j])
                    img[i, j] = np.subtract(img[i, j], model.predict(x))
                    img[i, j] = np.transpose(scaler.fit_transform(np.transpose([img[i, j]])))[0]
        self.normalizedImage = img

    def findDominant(self):
        # Creates ENVI Object and reads the image at the given path
        ei = Envi.EnviImage()
        ei.Read(self.PATH, True, False, False)

        # Decimates spectrum based on optional input(Default is 1, does not affect data)
        ei.DecimateSpectrum(self.decimate_factor)

        # Saves original image for later use in display
        self.origImage = ei.Pixels
        self.WAVELENGTHS = ei.wavelength
        # Normalizes image for shape
        self.normalize()
        img = self.normalizedImage
        # reshapes image into 2D array shape (x*y, z)
        img = img.reshape((ei.Pixels.shape[0] * ei.Pixels.shape[1], ei.Pixels.shape[2]))

        #reduces dimensions through pca


        # Kmeans clustering
        # Uses elbow method to calculate the optimal K clusters (Unless override by user, where cluster_override != 0)


        if self.PCAON:
            print("\nReducing Dimensions with PCA")
            pca = PCA()
            pca.fit(img)
            # print(pca.explained_variance_ratio_)
            var_sum = 0
            for x in range(len(pca.explained_variance_ratio_)):
                if pca.explained_variance_ratio_[x] > 0.06 or self.PCADIMENSIONS < 3:
                    var_sum += pca.explained_variance_ratio_[x]
                    self.PCADIMENSIONS += 1
                else:
                    break
            print(f"{var_sum * 100}% of signal represented with {self.PCADIMENSIONS} pixel dimensions\n")
            pca = PCA(n_components=self.PCADIMENSIONS)
            pca.fit(img)
            #print(pca.explained_variance_ratio_)
            img = pca.transform(img)

            # writing new image data to csv
            #np.savetxt("pca_image.csv", img, delimiter=',')

        print("Finding optimal number of clusters")
        self.IMAGE = img
        # cluster enumeration algorithms
        if self.cluster_override != 0:
            self.CLUSTERS = self.cluster_override
        elif self.CLUSTER_ENUM == 'elbow':
            #elbow method
            self.CLUSTERS = Elbow.elbowMethod(self, img)
            print(f"Elbow at {self.CLUSTERS} clusters")
        elif self.CLUSTER_ENUM == 'silhouette':
            #silhouette method
            self.CLUSTERS = Silhouette(range(2, 8)).silhouetteMethod(img)
            print(f"Highest Avg Silhouette score at {self.CLUSTERS} clusters")
        elif self.CLUSTER_ENUM == 'bic':
            #Custom BIC
            self.CLUSTERS = BIC(range(1, 15)).customBIC(img, self.normalizedImage.reshape((self.normalizedImage.shape[0] * self.normalizedImage.shape[1], self.normalizedImage.shape[2])))
            print(f"Optimal BIC score at {self.CLUSTERS} clusters")
        elif self.CLUSTER_ENUM == 'aic':
            #AIC
            self.CLUSTERS = AIC(range(1, 15)).aicMethod(img)
            print(f"Lowest AIC score at {self.CLUSTERS} clusters")
        elif self.CLUSTER_ENUM == 'gap':
            #Gap Statistic
            self.CLUSTERS = Gap().getGap(KMeans(n_init=20),img)
            print(f"Lowest gap score at {self.CLUSTERS} clusters")
        elif self.CLUSTER_ENUM == 'ded':
            #DeD
            self.CLUSTERS = DeD_Enumerator(img).optimal_k(range(2, 11))
            print(f"Optimal number of DeD clusters: {self.CLUSTERS} clusters")

        print('\nRunning Clustering Algorithm...')
        return img

    def plot(self):
        points = self.origImage
        centroids = self.CENTROIDS
        k = self.CLUSTERS # the number of clusters in the image
        labels = self.LABELS
        # Temporary solution for color key. Require better method of creating differentiable colors
        # (Currently only selects colors from the colorChoices array)
        # If number of clusters > len(colorChoices) algorithm output would be wrong.
        # randomly generates a list of colors to apply to the clusters of the image
        colorChoices = [np.array(np.random.choice(range(256), size=3))/255 for i in range(k)]
        """colorChoices = [[1, 0, 0], [1, 1, 0], [0, 0.92, 1], [0.66, 0, 1], [1, 0.5, 0], [0.75, 1, 1], [0, 0.58, 1], [1, 0, 0.66],
                        [1, 0.83, 0], [0.42, 1, 0], [0.93, 0.73, 0.73], [0.73,0.84,0.93],[0.9,0.91,0.73],[0.86,0.73,0.93],[0.73,0.93,0.88],
                        [0.56,0.14,0.14],[0.14,0.38,0.56],[0.56,0.42,0.14],[0.42,0.14,0.56],[0.31,0.56,0.14],[0,0,0],[0.45,0.45,0.45],[0.8,0.8,0.8]]"""
 
        # Plots the wavenumber vs absorption graph for each centroid color coded
        for center in range(k):
            plt.close()
            plt.figure()
            plt.plot(self.WAVELENGTHS, self.CENTROIDS[center], color=colorChoices[center], marker="o",label="Center " + str(center + 1))
            plt.savefig("/content/Results/single/" + self.imageName + "_SeperateClusteredAbsorptionGraph0" + str(center+1) + ".png")
            plt.close()
        plt.figure()
        plt.ylabel('Absorption')
        plt.xlabel('Wave Number')
        plt.title(f"Wave Number vs Absorption For Each Center ({self.ALG})")
        for center in range(k):
            plt.plot(self.WAVELENGTHS, self.CENTROIDS[center], color=colorChoices[center], marker="o",
                     label="Center " + str(center + 1))
        plt.legend()
        plt.savefig(self.RESULT_PATH + self.imageName + "_ClusteredAbsorptionGraph.png")
        plt.legend()
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

        #df = pandas.read_csv(path)
        # print(df)
        # makes CSV file of the PCA-transformed image data in the given directory
        path1 = self.RESULT_PATH + self.imageName + '_pca' + '.csv'

        with open(path1, mode='w+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            i = 0
            for row in self.IMAGE:
                row = list(row)
                row.append(self.LABELS[i])
                i+=1
                writer.writerow(row)
