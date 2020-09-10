from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from math import sqrt
from scipy.spatial.distance import cdist
import threading
import queue
class Elbow:

    def __init__(self):
        return

    def elbowMethod(self, X):

        #One minitbatch kmean function
        def doKmean(q, k, X):
            kmeanModel = KMeans(n_clusters=k).fit(X)
            #saves output to queue as a tuplet
            #(distortion, intertia)
            q.put((sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[
                0], kmeanModel.inertia_))

        #start queue
        q = queue.Queue()
        distortions = []
        #inertia = Sum squared distance
        inertias = []
        #Iterates K clustering for different K values from 1 to 10 with threading
        threads = []
        K = range(1, 10)
        print("Testing optimal K-clusters...")
        #Creates thread for each k value
        tempK = []
        for k in K:
            tempK.append(k)
            t = threading.Thread(target = doKmean, args=(q,k,X))
            t.start()
            threads.append(t)

        #Join every thread in the threads list so main will wait until all threads are finished
        for t in threads:
            t.join()

        #Saves output from queue
        #Each thread returns tuplet (distortion, intertia)
        while not q.empty():
            thing = q.get()

            distortions.append(thing[0])
            inertias.append(thing[1])
        if inertias[0] < inertias[1]:
            tempInert = inertias[0]
            inertias[0] = inertias[1]
            inertias[1] = tempInert

            tempDist = distortions[0]
            distortions[0] = distortions[1]
            distortions[1] = tempDist

        x1, y1 = 2, inertias[0]
        x2, y2 = 20, inertias[len(inertias)-1]

        #calculates the distances and append it to distances list
        distances = []
        for i in range(len(inertias)):
            x0 = i+2
            y0 = inertias[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)

        #print(str(inertias))
        #print(distortions)
        return distances.index(max(distances)) + 2