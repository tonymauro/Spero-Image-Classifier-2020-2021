from kmeans import KMeans
from Output import graphInertias
import numpy as np
km = KMeans(3)
km.k, inertias = km.elbow(13, 80, "test.bmp")
km.train(30)

graphInertias(13, inertias)
km.outputGraph()
km.outputImage()