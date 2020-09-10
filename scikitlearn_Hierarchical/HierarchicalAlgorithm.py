import matplotlib.pyplot as plt
import numpy as np
import pandas
import csv
import webcolors
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scikitlearn_Hierarchical.ENVI_Files import Envi
from collections import Counter
from datetime import datetime
from elbowMethod import Elbow
import time
import sys


# Return how much time has passed since given time, rounded to 2 decimals as a string
def time_difference(last_time):
    return str(round(time.time() - last_time))


class Log(object):
    RESULT_FOLDER = None  # The folder where general results are stored (top-level)
    RESULT_PATH = None  # The path to where results for this run are stored
    NO_RESULTS = None

    def __init__(self, log_path, no_results_output, result_log_path):
        self.RESULT_FOLDER = log_path
        self.orgstdout = sys.stdout
        self.NO_RESULTS = no_results_output
        self.RESULT_PATH = result_log_path

    def flush(self):
        self.orgstdout.flush()

    # Called by sys.stdout.write, or print()
    # Used to write the console output to log files
    def write(self, msg):
        # Print calls write twice, once with the message, and once with an empty new line, which messes up the log,
        # so this ignores any newline only messages. This is a messy solution
        if msg == "\n":
            return

        self.orgstdout.write(msg)  # Print the output to the console, as we have overridden this functionality
        now = datetime.now()
        time_string = now.strftime("%H-%M-%S")  # Get the current time in 24hr format as string
        if os.path.isdir(self.RESULT_FOLDER):  # Ensure folder for latest-log.txt exists
            # mode 'a' will create thee file if it does not exist
            with open(self.RESULT_FOLDER + "latest-log.txt", mode='a') as log:
                log.write(time_string + ": " + str(msg))  # Write the current time and append the message
        if not self.NO_RESULTS and os.path.isdir(self.RESULT_PATH):  # Ensure we are storing results, and the dir exists
            with open(self.RESULT_PATH + "\\log.txt", mode='a') as log:
                log.write(time_string + ": " + str(msg))


# Runs this clustering algorithm from the command line with prompts/user input
def run_cml():
    text = input("Enter Image Path (Start with * to enter custom directory), "
                 "\ncluster override: -c#, n_neighbors override: -n#, "
                 "\nDecimate Factor: -d#, No Results: -nr: ").split()
    if text[0].startswith("*"):
        image = text[1:]
    else:
        image = "ENVI_Files\\Images\\" + text[0]

    clusters = 0
    n_neighbors = 0
    decimate = 1
    no_results = False

    for index in range(len(text) - 1):
        if text[index + 1].startswith("-c"):
            clusters = int(text[index + 1][2:])
        elif text[index + 1] == "-nr":
            no_results = True
        elif text[index + 1].startswith("-n"):
            n_neighbors = int(text[index + 1][2:])
        elif text[index + 1].startswith("-d"):
            decimate = int(text[index + 1][2:])

    hca = HierarchicalClusterAlgorithm(image, clusters, n_neighbors, decimate, no_results)


class HierarchicalClusterAlgorithm:
    CLUSTERS = None
    IMAGE = None
    LABELS = None
    WAVELENGTHS = None
    CENTROIDS = None
    PATH_TO_ENVI = None
    ENVI_NAME = None
    RESULT_FOLDER = "Results\\"  # The folder where general results are stored (top-level)
    RESULT_PATH = RESULT_FOLDER  # The path to where results for this run are stored
    nNEIGHBORS = None
    is_cluster_override = False
    is_n_neighbors_override = False
    NO_RESULTS = False
    DECIMATE = 1

    # Takes in the image path, and cluster/n_neighbors overrides > 0
    def __init__(self, image_file, cluster_override=0, n_neighbors_override=0, decimate_factor=1,
                 no_results_output=False):

        try:
            # Make the Results folder if it does not exist.
            # Necessary for the latest-log file to be created.
            if not os.path.isdir(self.RESULT_FOLDER):
                try:
                    os.makedirs(self.RESULT_FOLDER)
                except OSError:
                    print("COULD NOT MAKE RESULTS FOLDER, NO DATA WILL BE SAVED")

            self.IMAGE = image_file
            self.origImage = image_file
            self.PATH_TO_ENVI = image_file

            name = str(self.PATH_TO_ENVI).split("\\")
            name = str(name[len(name) - 1])
            self.ENVI_NAME = name

            # Clear the latest-log file if it exists
            if os.path.isfile(self.RESULT_FOLDER + "latest-log.txt"):
                open(self.RESULT_FOLDER + "latest-log.txt", mode='w')

            if no_results_output:
                mkdir_result = "Not saving results"
                self.NO_RESULTS = True
            else:
                mkdir_result = self.make_result_dir()

            sys.stdout = Log(self.RESULT_FOLDER, self.NO_RESULTS, self.RESULT_PATH)

            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%m-%d-%Y %H-%M-%S")
            self.log("Starting at: " + dt_string + "\nLogging is Hr/Min/Sec (24hrs)\n")
            self.log(mkdir_result)

            if cluster_override > 0:
                self.is_cluster_override = True
                self.CLUSTERS = cluster_override
                self.log("Clusters count overridden to be: " + str(cluster_override))

            if n_neighbors_override > 0:
                self.is_n_neighbors_override = True
                self.nNEIGHBORS = n_neighbors_override
                self.log("n_neighbors count overridden to be: " + str(n_neighbors_override))

            if decimate_factor > 1:
                self.DECIMATE = int(decimate_factor)
                self.log("Set Decimate Factor to be: " + str(decimate_factor))

            total_time = time.time()
            self.cluster()
            if not self.NO_RESULTS:
                self.make_csv()
            self.log("Clustering complete, took: " + time_difference(total_time) + "s")
            self.plot()
        except Exception as e:
            sys.stdout.write("A fatal exception occurred, Error:\n%s" % e)

    def log(self, message):
        sys.stdout.write(message + "\n")

    def make_result_dir(self):
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m-%d-%Y %H-%M-%S")

        temp_path = self.RESULT_FOLDER + self.ENVI_NAME + " " + dt_string
        if os.path.isdir(temp_path):
            name_index = 1
            while os.path.isdir(temp_path + str(name_index)):
                try:
                    os.makedirs(temp_path + str(name_index))
                except OSError:
                    return "Failed to create duplicate directory for files"
                else:
                    temp_path += name_index
                    self.RESULT_PATH = temp_path
                    return "Duplicate directory created for files: " + self.RESULT_PATH
        else:
            try:
                os.makedirs(temp_path)
            except OSError:
                return "Failed to create directory for files"
            else:
                self.RESULT_PATH = temp_path
                return "Directory created for files: " + self.RESULT_PATH

    def cluster(self):
        """
        #read jpg, png or jfif image using cv2
        img = cv2.imread(self.IMAGE)
        #convert from BGR to RGB numpy arrays
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        """

        ei = Envi.EnviImage()
        ei.Read(self.IMAGE, True, False, False)

        plt.figure()
        plt.imshow(ei.Pixels[:, :, 2])
        plt.title("Original Image")

        if not self.NO_RESULTS:
            plt.savefig(self.RESULT_PATH + "\\" + self.ENVI_NAME + "_OriginalImage.png")

        if self.DECIMATE > 1:
            NL = np.array(ei.wavelength, copy=False).shape[-1]
            if self.DECIMATE <= NL // 2:
                self.log("Decimating... Length before: " + str(len(ei.wavelength)))
                decimate_result = ei.DecimateSpectrum(self.DECIMATE)
                self.log("Decimating complete, Length now: " + str(len(ei.wavelength)))
                if not decimate_result:
                    self.log("An error occurred while decimating")
                else:
                    plt.figure()
                    plt.imshow(ei.Pixels[:, :, 2])
                    plt.title("Decimated Image")
                    if not self.NO_RESULTS:
                        plt.savefig(self.RESULT_PATH + "\\" + self.ENVI_NAME + "_DecimatedImage.png")
            else:
                self.log("The decimate factor is too large, must be equal to or less than: " + str(NL // 2))

        self.origImage = ei.Pixels
        self.WAVELENGTHS = ei.wavelength

        img = []

        # reshapes image into 2D array shape (x*y, z)
        # Hierarchical only takes in 2D arrays
        for y in range(ei.Pixels.shape[1]):
            for x in range(ei.Pixels.shape[0]):
                img.append(ei.Pixels[x, y, :])
        img = np.array(img)

        # img = img.reshape((ei.Pixels.shape[0] * ei.Pixels.shape[1], ei.Pixels.shape[2]))
        self.IMAGE = img

        # Hierarchical clustering

        # get cluster amount with kmeans
        if self.is_cluster_override:
            self.log("Clusters overridden, skipping elbow method and using: " + str(self.CLUSTERS) + " clusters")
        else:
            st = time.time()
            self.log("Starting Elbow Method...")
            self.CLUSTERS = Elbow.elbowMethod(self, self.IMAGE)
            self.log("Elbow Method complete, took: " + str(round(time.time() - st, 2)) + "s")
            self.log("Final cluster size: " + str(self.CLUSTERS))

        # sets the connectivity, too resource intensive without
        st = time.time()
        self.log("Creating kneighbors_graph...")

        if self.is_n_neighbors_override:
            self.log("n_neighbors overridden, using: " + str(self.nNEIGHBORS))
            n_neighbors = self.nNEIGHBORS
        else:
            # Source for n_neighbors = clusters squared
            # I recommend searching for a better way to do this, or at least make it configurable by the user
            # https://bioinformatics.stackexchange.com/questions/4248/how-to-decide-number-of-neighbors-and-resolution-for-louvain-clustering
            n_neighbors = pow(self.CLUSTERS, 2)

        knn_graph = kneighbors_graph(img, n_neighbors, include_self=False)
        self.log("kneighbors_graph created, n_neighbors: " + str(n_neighbors) + ", took: " + time_difference(st) + "s")

        self.log("Starting Hierarchical Clustering (This may take a while)...")

        # set up the clustering with its parameters
        st = time.time()
        hierarchical = AgglomerativeClustering(n_clusters=self.CLUSTERS, linkage='ward', connectivity=knn_graph)
        hierarchical.fit(img)  # do the clustering
        self.log("Hierarchical Clustering complete, took: " + time_difference(st) + "s")

        self.LABELS = hierarchical.labels_
        self.make_avg_labels()

    def closest_colour(self, requested_colour):
        min_colours = {}
        for key, name in webcolors.css3_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    def plot(self):
        self.log("Plotting...")
        points = self.origImage
        k = self.CLUSTERS
        labels = self.LABELS
        color_key = []
        color_choices = [[0, 1, 0], [1, 0, 0], [0, .5, 1], [0, .5, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 0, 1],
                         [.5, .5, .5], [.5, .5, 1]]
        for center in range(k):
            # randColor = [rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)]
            rand_color = color_choices[center]
            color_key.append(rand_color)
        c = 0
        new_img = np.zeros((points.shape[0], points.shape[1], 3))
        for y in range(new_img.shape[1]):
            for x in range(new_img.shape[0]):
                new_img[x, y] = color_key[labels[c]]
                c += 1

        # Plots the 3D graph using R G B list collected from above and use colors from the clusters list
        plt.figure()
        plt.imshow(new_img)
        plt.title("Clustered Image")

        if not self.NO_RESULTS:
            plt.savefig(self.RESULT_PATH + "\\" + self.ENVI_NAME + "_ClusteredImage.png")

        plt.figure()
        plt.ylabel('Absorption')
        plt.xlabel('Wavenumbers')
        plt.title("Wavenumbers vs Absorption For Each Center")
        for x in range(k):
            temp_color = [rgb * 255 for rgb in color_choices[x]]
            self.log(self.closest_colour(temp_color) + ": " + str(self.CENTROIDS[x]))
            plt.plot(self.WAVELENGTHS, self.CENTROIDS[x], color=color_choices[x], marker="o",
                     label="Center " + str(x + 1))
        plt.legend()

        if not self.NO_RESULTS:
            plt.savefig(self.RESULT_PATH + "\\" + self.ENVI_NAME + "_ClusteredAbsorptionGraph.png")

        plt.show()

    # Return a 2D list of the average wavelengths for each cluster
    def make_avg_labels(self):

        st = time.time()
        self.log("Calculating cluster averages...")

        # Get amount of wavelengths per pixel
        wavelengths = len(self.IMAGE[0])

        # Create list to store the avg values for each cluster, for each wavelength
        avg_labels = np.zeros((self.CLUSTERS, wavelengths))

        index = 0  # Store current position in image pixels
        # Loop through the label list
        for label in self.LABELS:
            for x in range(wavelengths):
                # Add the corresponding IMAGE pixel values to the sum
                avg_labels[label][x] += self.IMAGE[index][x]
            index += 1

        # Count the occurrences of each label, and store them in a list
        x = Counter(sorted(self.LABELS))
        values = np.array(list(x.values()))

        # Divide each sum by its label total
        for label in range(self.CLUSTERS):
            avg_labels[label] /= values[label]

        # Round to 2 decimals
        avg_labels = avg_labels.round(2)
        self.CENTROIDS = avg_labels
        self.log("Cluster averages calculated, took: " + time_difference(st) + "s")

    def make_csv(self):
        csv_file_path = self.RESULT_PATH + "\\" + self.ENVI_NAME + ".csv"

        with open(csv_file_path, mode='w+', newline='') as csvFile:
            fieldnames = ["Wavenumber"]
            centers = self.CENTROIDS
            for x in range(len(centers)):
                fieldnames.append("Cluster " + str(x + 1))
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            writer.writeheader()
            for wl in self.WAVELENGTHS:
                temp_dict = {"Wavenumber": wl}
                for x in range(len(centers)):
                    temp_dict["Cluster " + str(x + 1)] = centers[x][self.WAVELENGTHS.index(wl)]
                writer.writerow(temp_dict)
        df = pandas.read_csv(csv_file_path)
        self.log("\n" + str(df))


run_cml()
