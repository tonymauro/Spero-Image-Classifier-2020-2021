from Input import get_arr
from Output import showPoints
from Output import renderImage
from Utility import updatePoint
from Utility import finishCheck
from Utility import updateCentroid
from Utility import getInertia
import random
import math
import numpy as np

import matplotlib.pyplot as plt

class KMeans:
  def __init__(self, dimensions):
    self.p = np.zeros(0)
    self.height = 0
    self.width = 0
    self.dimensions = dimensions
    self.points = np.zeros(0)
    self.centroids = np.zeros(0)
    self.k = 2
    self.maxIterations = 0
    self.centroids = np.zeros(0)
    self.pointLabels = np.zeros(0)

  def initialize(self, inputPath):
    self.p = get_arr(inputPath)
    self.height = len(self.p)
    self.width = len(self.p[0])
    tempPoints = np.zeros((self.dimensions,(self.width * self.height)))
    for i in range(len(self.p)):
      for j in range(len(self.p[0])):
        for k in range(self.dimensions):
          tempPoints[k][i * self.width + j] = self.p[i][j][k]
    self.points = tempPoints.T
    self.pointLabels = [None] * self.width * self.height
    print('initialized')
  def train(self, maxIterations):
    tempCentroids = np.zeros((self.dimensions, self.k))
    for i in range(self.dimensions):
      for j in range(self.k):
        tempCentroids[i][j] = random.random() * 255
    self.centroids = tempCentroids.T
    self.pointLabels = [None] * self.width * self.height
    iter = 0
    oldCentroids = np.zeros((self.k, self.dimensions))
    while iter < maxIterations:
      #Set labels relating each point to the closest centroid/ their cluster
      for i in range(len(self.points)):
        updatePoint(self.points[i], self.centroids, i, self.pointLabels)
      numberReset = 0
      for i in range(len(self.centroids)):
        potentialCentroid, reset = updateCentroid(self.centroids, self.points, i, self.pointLabels, self.dimensions)
        if not reset:
          self.centroids[i] = potentialCentroid
        else:
          numberReset = numberReset + 1
      if numberReset is not 0:
        print('reset', numberReset, 'centroids')
      if finishCheck(self.centroids, oldCentroids):
        print('finished')
        break
      oldCentroids = self.centroids.copy()
      iter += 1
      print(iter)
    return getInertia(self.points, self.pointLabels, self.centroids)
  def outputGraph(self):
    showPoints(self.points, self.centroids, self.k, self.pointLabels)
  def outputImage(self):
    renderImage(self.centroids, self.pointLabels, self.k, self.width, self.height)
  def elbow(self, maxK, maxIterations, inputPath):
    inertias = np.zeros(maxK - 1)
    distances = np.zeros(maxK - 1)
    #find all inertias
    for i in range(len(inertias)):
      self.k = i + 2
      self.initialize(inputPath)
      inertias[i] = self.train(maxIterations)
    print('found inertias')
    #use line-distance formula to find best cluster
    for i in range(len(inertias)):
      x0, y0 = i + 2, inertias[i]
      x1, y1 = 2, inertias[0]
      x2, y2 = maxK, inertias[len(inertias) - 1]

      numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
      denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
      distances[i] = numerator / denominator

    print(inertias)
    print('optimal k', distances.argmax() + 2)
    return distances.argmax() + 2, inertias



