import time
import sys
# insert at 1, 0 is the script path (or '' in REPL)
print(__file__)
sys.path.insert(1, __file__[:-8]+ '/ENVI_FILES')
import Envi

import matplotlib.pyplot as plt

yesList = ["yes", "y"]
answer = 'y'
pm = __import__("KMeans")
ei = Envi.EnviImage()
while answer in yesList:
    image = input("Enter file name: ")
    #Applies MiniKmeans
    start_time = time.time()
    dominant = pm.DominantColors(image)
    dominant.findDominant()
    # Shows elapsed time
    elapsed_time = time.time() - start_time
    print("Run time: " + str(elapsed_time))
    #Graphs RGB values in accordance to clusters found
    print("Clusters formed: ", dominant.CLUSTERS)
    answer = input("Cluster another file? (y/n) : ")

print("Ending task.")



