import numpy as np
import random
import collections
def updatePoint(point, centroids, index, pointLabels):
  pointLabels[index] = np.argmin(((point[:,None] - centroids.T)**2).sum(axis=0))
def updateCentroid(centroids, points, index, pointLabels, dimensions):
  indices = list()
  resets = 0
  for i in range(len(pointLabels)):
    if pointLabels[i] == index:
      indices.append(i)
  if len(indices) == 0:
    resets = resets + 1
    setRandomPos(centroids, index, dimensions)
    return np.zeros((dimensions)), True
  return np.divide(points[indices].sum(axis = 0), len(indices)), False
def getInertia(points, pointLabels, centroids):
  inertia = 0
  for i in range(len(points)):
    inertia += ((points[i] - centroids[pointLabels[i]])**2).sum()
  print('inertia',inertia)
  return inertia
def setRandomPos(centroids, index, dimensions):
  newPos = [0] * dimensions
  for i in range(dimensions):
    newPos[i] = random.random() * 255
  centroids[index] = newPos
def finishCheck(centroids, oldCentroids):
  return np.array_equal(centroids, oldCentroids)
