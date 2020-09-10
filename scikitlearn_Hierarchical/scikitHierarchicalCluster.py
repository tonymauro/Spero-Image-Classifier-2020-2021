import numpy as np
import cv2
import time as time
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from vispy import app, visuals, scene  # Requires PyQt (using v5)
from PIL import Image  # Pillow, for 2D image manipulation


class HierarchicalCluster:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    IMAGE_SIZE = None

    def __init__(self, image_, clusters):
        self.IMAGE = image_
        self.CLUSTERS = clusters

    # Returns the average rgb value of all provided pixels as an array
    def get_avg_pix(self, pixels):
        # initialize variables to hold the totals
        red = 0
        green = 0
        blue = 0

        # iterate over each pixel
        for x in pixels:  # x is an array of a pixel's rgb: [r, g, b]
            # increment each total by this pixel's value
            red += x[0]
            green += x[1]
            blue += x[2]

        # calculate the average by dividing by the total amount of pixels
        average = (red / len(pixels), green / len(pixels), blue / len(pixels))
        return average

    # Create a color to represent each label and populate COLORS with them
    def make_colors(self):

        print("Making Colors...")

        st = time.time()  # store the current time

        # Code to generate random colors
        # randcolor = randomcolor.RandomColor()
        # self.COLORS = []
        # for i in range(self.CLUSTERS):
        #     self.COLORS.append(randcolor.generate()[0])

        colors = []  # create an array to store all pixel rgb values, separated into different arrays by labels

        # populate colors with the same amount of empty arrays as CLUSTERS
        i = 0  # index
        while i < self.CLUSTERS:
            colors.append([])
            i += 1

        # populate colors with pixels, separated by label
        for label, pix in zip(self.LABELS, self.IMAGE):
            colors[label].append(pix)

        avg_colors = []  # array to store the final colors, will be of length CLUSTERS

        # populate avg_colors by calculating the average of the pixel arrays in colors
        for label_pixels in colors:
            avg_colors.append(self.get_avg_pix(label_pixels))

        self.COLORS = avg_colors  # assign the COLORS variable
        print("COLORS: " + str(self.COLORS))  # print the colors to the console

        print("Colors Made, Took: ", time.time() - st)  # print the time taken to compute the colors

    # Loads and returns the image specified by IMAGE
    def read_image(self):
        # read image
        #cv2 returns an array
        img = cv2.imread(self.IMAGE, cv2.IMREAD_COLOR)

        # convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # This sets the length and width of the image into IMAGE_SIZE
        self.IMAGE_SIZE = (img.shape[1], img.shape[0])
        return img

    # Method that does the clustering
    def cluster(self):
        img = self.read_image()  # read in the image

        # reshape the img size to be its length * width (all the pixels), with [r,g,b] for each pixel
        img = img.reshape((self.IMAGE_SIZE[0] * self.IMAGE_SIZE[1], 3))
        # img = np.resize(img, (12000, 3))  # Cuts off part of the image to be less resource intensive

        self.IMAGE = img  # sets IMAGE so it can be used in the display

        print(img.shape)  # prints the dimensions of the image for debugging


        # begin clustering
        print("\nClustering...")
        st = time.time()  # save the current time

        # connectivity = grid_to_graph(*img.shape)  # old connectivity code from original example

        # sets the connectivity, too resource intensive without
        knn_graph = kneighbors_graph(img, 30, include_self=False)

        # set up the clustering with its parameters
        hierarchical = AgglomerativeClustering(n_clusters=self.CLUSTERS, linkage='ward', connectivity=knn_graph)
        hierarchical.fit(img)  # do the clustering

        print("Cluster Complete, Took: ", time.time() - st)  # print the time taken to cluster

        # set LABELS with the result of the clustering. Labels are the assigned cluster for each pixel
        self.LABELS = hierarchical.labels_

        # create the color array based on the found labels
        self.make_colors()

    # create 2D representation of the cluster
    def imageplot(self):  # not working
        img = Image.new('RGB', self.IMAGE_SIZE)

        colors = []
        # i = 0
        # for label, pix in zip(self.LABELS, self.IMAGE):
        #     colors.append(hex_to_rgb(self.COLORS[label]))
        #     i += 1

        i = 0
        for x in range(self.IMAGE_SIZE[0]):
            for y in range(self.IMAGE_SIZE[1]):
                img.putpixel((x, y), colors[i])
                i += 1

        img.show()

    # creates a 3D scatterplot of the clustering
    def scatterplot(self):

        print("Creating 3D Scatterplot...")
        st = time.time()  # store the current time

        # build the visuals
        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)

        # build canvas
        canvas = scene.SceneCanvas(keys='interactive', show=True)

        # Add a ViewBox to let the user zoom/rotate
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 45
        view.camera.distance = 1000
        # TODO center camera more to the bottom left

        # initialize colors, which is used to set the colors of the points
        colors = np.ones((len(self.IMAGE), 3), dtype=np.float32)

        # loop through the pixels and labels to find the rgb values
        # vispy want rgb on a scale from 0-1, so we divide by 255
        i = 0  # index
        for label, pix in zip(self.LABELS, self.IMAGE):
            # colors will contain an rgb array for each point
            colors[i] = (self.COLORS[label][0]/255, self.COLORS[label][1]/255, self.COLORS[label][2]/255)
            i += 1

        # plot
        p1 = Scatter3D(parent=view.scene)
        # set the rendering parameters
        p1.set_gl_state('opaque', blend=True, depth_test=True)
        # set the point data. Positions are set to the IMAGE rgb + colors are from colors array
        p1.set_data(self.IMAGE, face_color=colors, symbol='o', size=10, edge_width=0)

        # Add axis
        # TODO labels do not display
        xax = scene.Axis(pos=[[0, 0], [500, 0]], tick_direction=(0, -1), axis_label='R', axis_color='r', tick_color='r',
                         text_color='r', font_size=160, parent=view.scene)
        yax = scene.Axis(pos=[[0, 0], [0, 500]], tick_direction=(-1, 0), axis_label='G', axis_color='g', tick_color='g',
                         text_color='g', font_size=160, parent=view.scene)

        zax = scene.Axis(pos=[[0, 0], [-500, 0]], tick_direction=(0, -1), axis_label='B', axis_color='b',
                         tick_color='b', text_color='b', font_size=160, parent=view.scene)
        zax.transform = scene.transforms.MatrixTransform()  # its actually an inverted x-axis
        zax.transform.rotate(90, (0, 1, 0))  # rotate cw around y-axis
        zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

        print("3D Scatterplot Completed, Took: ", time.time() - st)  # print the time taken to make the scatterplot

        app.run()  # display the scatterplot


image = "1080image.jpg"
dominant = HierarchicalCluster(image, 6)
dominant.cluster()
dominant.scatterplot()
# dominant.read_image()
# dominant.imageplot()
