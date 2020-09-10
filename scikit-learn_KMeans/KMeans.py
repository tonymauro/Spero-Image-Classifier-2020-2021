import matplotlib.pyplot as plt
import os
import pandas
import webcolors
import csv
from datetime import datetime
from sklearn.cluster import KMeans
from elbowMethod import Elbow
import numpy as np

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, __file__ + 'ENVI_FILES')
import Envi
from mpl_toolkits.mplot3d import axes3d, Axes3D
class DominantColors:
    CLUSTERS = None
    IMAGE = None
    CENTROIDS = None
    LABELS = None
    WAVELENGTHS = None

    def __init__(self, image):
        self.IMAGE = image
        self.origImage = image
        self.PATH = image
        self.RESULT_FOLDER = "Result\\"
        self.make_result_dir()
        self.RESULT_PATH += "\\"
    def closest_colour(self, requested_colour):
        min_colours = {}
        for key, name in webcolors.css3_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour
            [1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    def make_result_dir(self):
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m-%d-%Y %H-%M-%S")

        temp_path = self.RESULT_FOLDER + self.PATH + " " + dt_string
        if os.path.isdir(temp_path):
            name_index = 1
            while os.path.isdir(temp_path + str(name_index)):
                try:
                    os.makedirs(temp_path + str(name_index))
                except OSError:
                    return
                else:
                    temp_path += name_index
                    self.RESULT_PATH = temp_path
                    return
        else:
            try:
                os.makedirs(temp_path)
            except OSError:
                return
            else:
                self.RESULT_PATH = temp_path
                pass

    def findDominant(self):
        # To be revised into reading ENVI images
        '''
        #read jpg, png or jfif image using cv2
        img = cv2.imread(self.IMAGE)
        #convert from BGR to RGB numpy arrays
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        '''

        ei = Envi.EnviImage()
        ei.Read(self.IMAGE, True, False, False)

        # plt.imshow(np.array(ei.Pixels, np.dtype("int32")))
        self.origImage = ei.Pixels
        self.WAVELENGTHS = ei.wavelength
        print(len(self.WAVELENGTHS))
        # reshapes image into 2D array shape (x*y, z)
        # Kmeans only takes in 2D arrays
        img = np.array(ei.Pixels)
        img = img.reshape((ei.Pixels.shape[0] * ei.Pixels.shape[1], ei.Pixels.shape[2]))
        self.IMAGE = img

        # Kmeans clustering
        # Uses elbow method to calculate the optimal K clusters
        self.CLUSTERS = Elbow.elbowMethod(self, self.IMAGE)
        kmeans = KMeans(n_clusters=self.CLUSTERS, n_init=20)
        kmeans.fit(img)

        # Centroids are saved
        self.CENTROIDS = kmeans.cluster_centers_
        self.LABELS = kmeans.labels_
        self.plot()
        # Returns centers as XD arrays
        return self.CENTROIDS

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def plot(self):
        points = self.origImage
        print(points.shape)
        centroids = self.CENTROIDS
        print(centroids.shape)
        k = self.CLUSTERS
        labels = self.LABELS
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

        plt.figure()
        plt.imshow(newImg)
        plt.savefig(self.RESULT_PATH + self.PATH + "_ClusteredImage.png")
        plt.figure()
        plt.ylabel('Absorption')
        plt.xlabel('Wave Number')
        plt.title("Wave Number vs Absorption For Each Center")
        patches = []
        for x in range(k):
            tempColor = [rgb * 255 for rgb in colorChoices[x]]
            print(self.closest_colour(tempColor) + ": " + str(self.CENTROIDS[x]))
            plt.plot(self.WAVELENGTHS, self.CENTROIDS[x], color=colorChoices[x], marker="o",
                     label="Center " + str(x + 1))
        plt.legend()
        plt.savefig(self.RESULT_PATH + self.PATH + "_ClusteredAbsorptionGraph.png")
        self.makeCSV()

    def makeCSV(self):
        path = self.RESULT_PATH + self.PATH + '.csv'
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