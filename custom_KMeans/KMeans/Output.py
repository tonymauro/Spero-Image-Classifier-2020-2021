from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
def showPoints(points, centroids, k, pointLabels):
    r = list()
    g = list()
    b = list()
    clusters = list()
    for i in range(len(points)):
        r.append(points[i][0])
        g.append(points[i][1])
        b.append(points[i][2])
        clusters.append(pointLabels[i])
    for i in range(len(centroids)):
        r.append(centroids[i][0])
        g.append(centroids[i][1])
        b.append(centroids[i][2])
        clusters.append(k + 1)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(r, g, b, c=clusters, cmap='Accent', marker=',');
    print(centroids, k, pointLabels)
    plt.show()
def graphInertias(maxK, inertias):
    xs = np.arange(2, maxK + 1)
    plt.plot(xs, inertias)
    plt.show()
def renderImage(centroids,pointLabels, k, width, height):
    data = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width * height):
        data[int(i/width)][int(i%width)] = centroids[pointLabels[i]]
        #print(data[int(i/height)][int(i - int(i/width)* width)])
    img = Image.fromarray(data, 'RGB')
    img.show()
