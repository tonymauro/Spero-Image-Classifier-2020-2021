A prototype for the Spero Image Classifier

Overrides for GUI:
Custom Clusters - Allows the users to override the number of clusters to run the algorithm by in order to skip the elbow method
Decimation Factor - An option for the user to reduce the the spectrum by a certain factor to reduce run time(Efficient for large ENVI,
not suggested for smaller ENVI images)
Save Directory - User can choose where the results will be saved (Default is CompiledAlgorithms/Result)

Elbow Method - Determines optimal amount of clusters by running the clustering algorithm multiple times with different clusters.
Uses multithreading to speed up process by running all tests at the same time.

GUI Specific Pixel Spectrum Feature - Currently not accurate 100% of the time due to image scaling from the GUI output. Need to find a better
solution than recalculating resize scale(Due to rounding).

Required Libraries:
matplotlib
pyQt5
imageio
numpy
pandas
pyqtgraph
sklearn
scipy
webcolors
