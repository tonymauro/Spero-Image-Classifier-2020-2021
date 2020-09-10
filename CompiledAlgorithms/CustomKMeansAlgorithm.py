import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from CompiledAlgorithms.ENVI_Files import Envi
from CompiledAlgorithms.elbowMethod import Elbow
import pandas
import csv


class KMeans:
    def __init__(self):
        self.path = 0
        self.name = 0
        self.result = 0
        self.ei3 = Envi.EnviImage()
        self.origImage = 0
        self.wavelength = 0
        self.p = np.zeros(0)
        self.height = 0
        self.width = 0
        self.dimensions = 0
        self.points = np.zeros(0)
        self.centroids = np.zeros(0)
        self.k = 6
        self.maxIterations = 0
        self.centroids = np.zeros(0)
        self.pointLabels = np.zeros(0)
        self.inertias = list()
        self.cap = 0

    def initialize(self, path, decimation, result_folder, filename):
        self.path = path
        self.ei3.Read(path, True, False, False)
        self.ei3.DecimateSpectrum(decimation)
        self.wavelength = self.ei3.wavelength
        self.p = self.ei3.Pixels
        self.dimensions = len(self.p[0][0])
        self.height = len(self.p)
        self.width = len(self.p[0])
        self.result = result_folder
        self.name = filename
        tempPoints = np.zeros((self.dimensions, (self.width * self.height)))
        for i in range(len(self.p)):
            for j in range(len(self.p[0])):
                for k in range(self.dimensions):
                    if (self.ei3.Pixels[i][j][k] > self.cap):
                        self.cap = self.ei3.Pixels[i][j][k]
                    tempPoints[k][i * self.width + j] = self.p[i][j][k]
        self.points = tempPoints.T
        self.pointLabels = [None] * self.width * self.height
        print('initialized')

    def updatePoint(self, point, centroids, index, pointLabels):
        pointLabels[index] = np.argmin(((point[:, None] - centroids.T) ** 2).sum(axis=0))

    def updateCentroid(self, centroids, points, index, pointLabels, dimensions):
        indices = list()
        resets = 0
        for i in range(len(pointLabels)):
            if pointLabels[i] == index:
                indices.append(i)
        if len(indices) == 0:
            resets = resets + 1
            self.setRandomPos(centroids, index, dimensions)
            return np.zeros((dimensions)), True
        return np.divide(points[indices].sum(axis=0), len(indices)), False

    def getInertia(self, points, pointLabels, centroids):
        inertia = 0
        for i in range(len(points)):
            inertia += ((points[i] - centroids[pointLabels[i]]) ** 2).sum()
        print('inertia', inertia)
        return inertia

    def setRandomPos(self, centroids, index, dimensions):
        ran_h = int(self.height * random.random())
        ran_w = int(self.width * random.random())
        centroids[index] = self.p[ran_h][ran_w]

    def finishCheck(self, centroids, oldCentroids):
        return np.array_equal(centroids, oldCentroids)

    def train(self, maxIterations):
        tempCentroids = np.zeros((self.dimensions, self.k))
        for i in range(self.dimensions):
            for j in range(self.k):
                tempCentroids[i][j] = random.random() * self.cap
        self.centroids = tempCentroids.T
        self.pointLabels = [None] * self.width * self.height
        iter = 0
        oldCentroids = np.zeros((self.k, self.dimensions))
        while iter < maxIterations:
            # Set labels relating each point to the closest centroid/ their cluster
            for i in range(len(self.points)):
                self.updatePoint(self.points[i], self.centroids, i, self.pointLabels)
            numberReset = 0
            for i in range(len(self.centroids)):
                potentialCentroid, reset = self.updateCentroid(self.centroids, self.points, i, self.pointLabels,
                                                               self.dimensions)
                if not reset:
                    self.centroids[i] = potentialCentroid
                else:
                    numberReset = numberReset + 1
            if numberReset is not 0:
                print('reset', numberReset, 'centroids')
            if self.finishCheck(self.centroids, oldCentroids):
                print('finished')
                break
            oldCentroids = self.centroids.copy()
            iter += 1
            print(oldCentroids)
            print('epoch', iter)
        return self.getInertia(self.points, self.pointLabels, self.centroids)

    def outputGraph(self):
        r = list()
        g = list()
        b = list()
        clusters = list()
        for i in range(len(self.points)):
            r.append(self.points[i][0])
            g.append(self.points[i][1])
            b.append(self.points[i][2])
            clusters.append(self.pointLabels[i])
        for i in range(len(self.centroids)):
            r.append(self.centroids[i][0])
            g.append(self.centroids[i][1])
            b.append(self.centroids[i][2])
            clusters.append(self.k + 1)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(r, g, b, c=clusters, cmap='Accent', marker=',')
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        print(self.centroids, self.k, self.pointLabels)
        plt.show()

    def outputImageAlternate(self):
        data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(self.width * self.height):
            data[int(i / self.width)][int(i % self.width)] = np.array(
                [self.pointLabels[i] * 30 % 255, self.pointLabels[i] * 60 % 255, self.pointLabels[i] * 90 % 255])
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(data)
        plt.savefig(self.result + "/" + self.name + "_ClusteredImage.png", bbox_inches="tight", pad_inches=0)

    # Not used, outputs too small image. Using Alternate instead
    def outputImage(self):
        data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(self.width * self.height):
            data[int(i / self.width)][int(i % self.width)] = np.array(
                [self.pointLabels[i] * 30 % 255, self.pointLabels[i] * 60 % 255, self.pointLabels[i] * 90 % 255])
        img = Image.fromarray(data, 'RGB')
        img.save(self.result + "/" + self.name + "_ClusteredImage.png", 'PNG')

    def elbow(self, maxK, maxIterations, inputPath, decimation, result_folder, filename):
        ei = Envi.EnviImage()
        ei.Read(inputPath, True, False, False)

        img = np.array(ei.Pixels)
        img = img.reshape((ei.Pixels.shape[0] * ei.Pixels.shape[1], ei.Pixels.shape[2]))
        kClusters = Elbow.elbowMethod(self, img)
        print(kClusters)
        return kClusters

    def graphInertias(self):
        xs = np.arange(2, len(self.inertias) + 2)
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.plot(xs, self.inertias)
        plt.show()

    def plot(self):
        points = self.p
        print(points.shape)
        centroids = self.centroids
        print(centroids.shape)
        k = self.k
        labels = self.pointLabels
        print(labels)
        colorKey = []
        colorChoices = [[0, 1, 0], [1, 0, 0], [0, 1, 1], [0.5, 0.5, 0], [1, 0, 1], [1, .5, 1], [1, 0, 1], [0, 0, 1],
                        [.5, .5, .5], [.5, .5, 1]]
        for center in range(k):
            # randColor = [rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)]
            randColor = colorChoices[center]
            colorKey.append(randColor)
        c = 0
        newImg = np.zeros((points.shape[0], points.shape[1], 3))
        for x in range(newImg.shape[0]):
            for y in range(newImg.shape[1]):
                newImg[x, y] = colorKey[labels[c]]
                c += 1
        # Plots the 3D graph using R G B list colected from above and use colors from the clusters list
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(newImg)
        plt.figure(frameon=False)
        plt.ylabel('Absorption')
        plt.xlabel('Wavelength')
        plt.title("Wavelength vs Absorption For Each Center")
        patches = []
        for x in range(k):
            tempColor = [rgb * 255 for rgb in colorChoices[x]]
            #   print(self.closest_colour(tempColor) + ": " + str(self.centroids[x]))
            plt.plot(self.wavelength, self.centroids[x], color=colorChoices[x], marker="o",
                     label="Center " + str(x + 1))
        plt.legend()
        plt.savefig(self.result + self.name + "_ClusteredAbsorptionGraph.png")

    def makeCSV(self):
        path = self.result + self.name + '.csv'
        with open(path, mode='w') as csvFile:
            fieldnames = ['Wavelength']
            for x in range(len(self.centroids)):
                fieldnames.append("Cluster " + str(x + 1))
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            writer.writeheader()
            for wl in self.wavelength:
                tempDict = {"Wavelength": wl}
                for x in range(len(self.centroids)):
                    tempDict["Cluster " + str(x + 1)] = self.centroids[x][self.wavelength.index(wl)]
                writer.writerow(tempDict)
        df = pandas.read_csv(path)
        print(df)
