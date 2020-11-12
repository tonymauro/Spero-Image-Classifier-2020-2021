from PyQt5.QtWidgets import QApplication, QHBoxLayout, QFormLayout, QLineEdit, QCheckBox, QWidget, QPushButton, \
    QSizePolicy, QMainWindow, QLabel, QGroupBox, QVBoxLayout, QFileDialog, QGridLayout, QComboBox
from PyQt5.QtGui import QPixmap, QIcon, QImage, QColor
import pyqtgraph as pg
import numpy as np
from CompiledAlgorithms import AlgorithmRunner as algoRunner
from CompiledAlgorithms.ENVI_Files import Envi
from datetime import datetime
import os


class MainWindow(QMainWindow):
    def __init__(self):
        # Sets up GUI window with default variables and window icon
        QMainWindow.__init__(self)
        self.setWindowTitle("Spero Classifier")
        self.setFixedSize(700, 200)
        self.setWindowIcon(QIcon('CompiledAlgorithms/DRS_Logo.jfif'))
        self.filePath = ""
        self.currentDir = "Not selected"
        self.init_Gui()
        self.show()

    def init_Gui(self):
        # Uses Grid layout, with two boxes horizontally
        # Calls these two functions to make the two boxes with the widgets needed
        self.makeUploadBox()
        self.makeManualCustomBox()

        # Main layout of the GUI is initialized and the two boxes created are added as widgets horizontally
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.uploadBox, 1, 0)
        mainLayout.addWidget(self.manualBox, 1, 1)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)

        # Set main layout to the widget
        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(mainLayout)

    def makeUploadBox(self):
        # Box for GUI objects related to files
        self.uploadBox = QGroupBox("ENVI File")
        self.uploadBox.setFixedSize(350, 200)
        self.uploadBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        #  will be the main layout for the box with hLayout inside the layout
        vLayout = QVBoxLayout()
        hLayout = QHBoxLayout()

        # Button layout for start and clear buttons
        hButtonLayout = QHBoxLayout()

        # File selector. currentDirText displays the currently selected file, Ok_Button opens file window for user to select
        # ENVI file
        self.currentDirText = QLabel("Currently selected file: None")
        self.OK_Button = QPushButton("Select ENVI Image")
        self.OK_Button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.OK_Button.clicked.connect(self.openFile)

        # Clear button linked to function clearAll to clear all selected files and settings
        self.clear_Button = QPushButton("Clear All")
        self.clear_Button.clicked.connect(self.clearAll)

        self.startButton = QPushButton("Start analysis")

        # Dropdown list for list of algorithms. Checks the currently selected algorithm
        # For suitable override options with self.checkDropDown
        self.DropDownList = QComboBox()
        self.DropDownList.addItem("Choose Clustering Algorithm")
        self.DropDownList.addItem("K-Means Clustering: Scikit-Learn")
        self.DropDownList.addItem("Gaussian Mixture Model: Scikit-Learn")
        self.DropDownList.addItem("Custom K-Means Clustering Algorithm")
        self.DropDownList.addItem("Hierarchical Clustering Algorithm: Scikit-Learn")
        self.DropDownList.addItem("Experimental")
        self.DropDownList.currentIndexChanged.connect(self.checkDropDown)

        # Adds folowing widgets to hLayout for file selection section
        hLayout.addWidget(self.currentDirText)
        hLayout.addWidget(self.OK_Button)
        # Adds following buttons to hButtonLayout
        hButtonLayout.addWidget(self.startButton)
        hButtonLayout.addWidget((self.clear_Button))

        # Adds hLayout, Drop Down List and hButtonLayout to vLayout
        vLayout.addLayout(hLayout)
        vLayout.addWidget(self.DropDownList)
        vLayout.addLayout(hButtonLayout)
        # Calls the startAlgorithm function when start button is clicked, enabled false
        # By default until user has selected an algorithm and an ENVI File
        self.startButton.clicked.connect(self.startAlgorithm)
        self.startButton.setEnabled(False)
        self.uploadBox.setLayout(vLayout)

    def makeManualCustomBox(self):
        # Box for manual overrides for certain algorithms
        # Using QFormLayout to make rows for each override along with their check marks
        self.manualBox = QGroupBox("Optional Manual Adjustments")
        self.manualBox.setFixedSize(350, 200)
        self.manualBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        manualLayout = QFormLayout()
        # All following override GUI except saveDir are disabled until
        # User checks the checkmark next to the option they want to override. Else
        # If unchecked the algorithm will not take the override.

        # Cluster override that allows users to select a number of clusters to run the algorithm by.
        # If no override, algorithm will use elbow method.
        customCluster = QLineEdit()
        clusterCheckBox = QCheckBox("Custom Clusters")
        clusterCheckBox.stateChanged.connect(self.stateChangeCluster)
        customCluster.setPlaceholderText("Enter Custom Clusters")
        customCluster.setEnabled(False)
        self.customCluster = customCluster
        self.clusterCheckBox = clusterCheckBox

        # Decimate override allows user to select decimation factor to reduce
        # the data by, not recommended for small data sets.
        decimateCheck = QCheckBox("Decimation Factor")
        DecimateText = QLineEdit()
        decimateCheck.stateChanged.connect(self.stateChangeDecimate)
        DecimateText.setEnabled(False)
        DecimateText.setPlaceholderText("Set Decimation Factor")
        self.decimateCheck = decimateCheck
        self.DecimateText = DecimateText

        # N-Neighbor override only applies to Hiearchal Clustering
        # Algorithm. Will not be enabled for other K-Means clustering
        neighborCheck = QCheckBox("Custom N Neighbors")
        neighborInput = QLineEdit()
        neighborCheck.setEnabled(False)
        neighborCheck.stateChanged.connect(self.stateChangeNeighbor)
        neighborInput.setEnabled(False)
        neighborInput.setPlaceholderText("Enter N Neighbors here")
        self.neighborCheck = neighborCheck
        self.neighborInput = neighborInput

        # Button allows user to select a directory where they want to save the results in
        # In case they want to review it for later use. Default directory will be in the Results
        # Folder in CompiledAlgorithms
        self.dirButton = QPushButton("Select Directory to Save Result Folder")
        self.dirButton.clicked.connect(self.getDir)
        self.currentDirLabel = QLabel(self.currentDir)
        self.currentDirConstantLabel = QLabel("Current Save Directory:")

        # Adds each override options along with their checkboxes to each row
        manualLayout.addRow(self.clusterCheckBox, self.customCluster)
        manualLayout.addRow(self.decimateCheck, self.DecimateText)
        manualLayout.addRow(self.neighborCheck, self.neighborInput)
        manualLayout.addRow(self.dirButton)
        manualLayout.addRow(self.currentDirConstantLabel)
        manualLayout.addRow(self.currentDirLabel)

        self.manualBox.setLayout(manualLayout)

    def clearAll(self):
        # Resets all current options to default.
        self.currentDir = "Not selected"
        self.filePath = ""
        self.currentDirText.setText("Currently selected file: None")
        self.DropDownList.setCurrentIndex(0)

        self.customCluster.setText("")
        self.customCluster.setEnabled(False)
        self.clusterCheckBox.setChecked(False)

        self.DecimateText.setText("")
        self.DecimateText.setEnabled(False)
        self.decimateCheck.setChecked(False)

        self.neighborInput.setText("")
        self.neighborInput.setEnabled(False)
        self.neighborCheck.setChecked(False)

        self.currentDirLabel.setText(self.currentDir)

    def openFile(self):
        #Opens a folder dir window for user to select the ENVI file
        filePath, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)")
        self.filePath = filePath
        name = filePath[self.filePath.rfind("/") + 1:len(filePath)]
        self.currentDirText.setText("Currently selected file: " + name)
        if self.DropDownList.currentText() != "Choose Clustering Algorithm":
            self.startButton.setEnabled(True)

    def getDir(self):
        # Opens new file window for user to select a directory to save the result files
        self.currentDir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.currentDirLabel.setText(self.currentDir)

    def stateChangeCluster(self):
        if self.clusterCheckBox.isChecked():
            self.customCluster.setEnabled(True)
        else:
            self.customCluster.setEnabled(False)

    def stateChangeDecimate(self):
        if self.decimateCheck.isChecked():
            self.DecimateText.setEnabled(True)
        else:
            self.DecimateText.setEnabled(False)

    def stateChangeNeighbor(self):
        if self.neighborCheck.isChecked():
            self.neighborInput.setEnabled(True)
        else:
            self.neighborInput.setEnabled(False)

    def stateChangeSelectDirectory(self):
        if self.saveDirCheck.isChecked():
            self.dirButton.setEnabled(True)
        else:
            self.dirButton.setEnabled(False)

    def checkDropDown(self):
        # Toggles override options for different clustering algorithms
        # Makes sure run button is not enabled when no algorithm and file is selected
        text = self.DropDownList.currentText()
        print(self.currentDirText.text())
        if text != "Choose Clustering Algorithm" and self.filePath:
            self.startButton.setEnabled(True)
        else:
            self.startButton.setEnabled(False)

        if text == "Hierarchical Clustering Algorithm: Scikit-Learn":
            self.neighborCheck.setEnabled(True)
            self.stateChangeNeighbor()
            return

        self.neighborCheck.setEnabled(False)
        self.neighborInput.setEnabled(False)

    def make_result_dir(self, imageName, currentDir):
        # Create the result folder at the given directory
        # Along with the time stamps
        # If no directory specified, creates the folder
        # at ...CompiledAlgorithms/Result
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m-%d-%Y %H-%M-%S")

        temp_path = currentDir + imageName + " " + dt_string
        print(temp_path)
        if os.path.isdir(temp_path):
            name_index = 1
            while os.path.isdir(temp_path + str(name_index)):
                try:
                    os.makedirs(temp_path + str(name_index))
                except OSError:
                    return
                else:
                    temp_path += name_index
                    return temp_path + "\\"
        else:
            try:
                os.makedirs(temp_path)
            except Exception as e:
                print(e)
                return
            else:
                return temp_path + "\\"
                pass

    def startAlgorithm(self):
        # Creates algorithmRunner containing all the algorithm functions
        # Creates the running window and hide the main window while the algorithm is running
        # Checks overrides and set default values if no override values selected
        # Checks the string of the selected algorithm and
        # call the corresponding algorithm with parameters from user input
        # After algorithm runs, algorithm will create a folder containing
        # the result files and GUI will access the folder
        # To display the output
        # After run GUI will create result window with the result files created by the algorithm
        runner = algoRunner.AlgorithmRunner()
        self.newWindow = Running()
        self.hide()
        self.newWindow.show()
        clusterOverride = 0
        nOverride = 0
        decimateOverride = 1
        if self.clusterCheckBox.isChecked():
            try:
                clusterOverride = int(self.customCluster.text())
            except:
                pass

        if self.neighborCheck.isChecked():
            try:
                nOverride = int(self.neighborInput.text())
            except:
                pass

        if self.decimateCheck.isChecked():
            try:
                decimateOverride = int(self.DecimateText.text())
            except:
                pass

        # Checks if user specified on a specific directory to save the result folder
        if self.currentDir == "Not selected":
            currentDir = str(__file__)[:-7] + "/Result/"
        else:
            currentDir = self.currentDir + "\\"

        # Get the name of the ENVI image
        filename = self.filePath[self.filePath.rfind("/") + 1:len(self.filePath)]

        # Make the result folder at the given directory
        currentDir = self.make_result_dir(filename, currentDir)

        try:
            if self.DropDownList.currentText() == "Custom K-Means Clustering Algorithm":
                runner.run_CustomKMeansAlgorithm(self.filePath, filename, currentDir, custom_clusters=clusterOverride,
                                                 decimation=decimateOverride, max_iterations=30)
            elif self.DropDownList.currentText() == "Hierarchical Clustering Algorithm: Scikit-Learn":
                runner.run_HierarchicalClusterAlgorithm(self.filePath, filename, currentDir,
                                                        cluster_override=clusterOverride, n_neighbors_override=nOverride,
                                                        decimate_factor=decimateOverride)
            elif self.DropDownList.currentText() == "K-Means Clustering: Scikit-Learn":
                runner.run_kMeansAlgorithm(self.filePath, filename, currentDir, cluster_override=clusterOverride,
                                           decimate_factor=decimateOverride)
            elif self.DropDownList.currentText() == "Gaussian Mixture Model: Scikit-Learn":
                runner.run_GMMAlgorithm(self.filePath, filename, currentDir, cluster_override=clusterOverride,
                                           decimate_factor=decimateOverride)
            elif self.DropDownList.currentText() == "Experimental":
                runner.run_EXPAlgorithm(self.filePath, filename, currentDir, cluster_override=clusterOverride,
                                           decimate_factor=decimateOverride)
            self.newWindow.close()
            self.show()
            self.resultWindow = Result(currentDir, filename, self.filePath)
            self.resultWindow.show()
        except Exception as e:
            print(e)
            try:
                self.newWindow.close()
                self.resultWindow.close()
            except:
                pass
            self.show()

        # access self.filePath and selected algorithm and call the corresponding algorithm function.


class Running(QMainWindow):
    def __init__(self):
        # Shows the user algorithm is running. Will not respond during the run if
        # file is large
        QMainWindow.__init__(self)
        self.setWindowTitle("Running Algorithm...")
        self.setWindowIcon(QIcon('CompiledAlgorithms/DRS_Logo.jfif'))
        self.setFixedSize(700, 200)
        layout = QVBoxLayout()
        self.text = QLabel("Running Algorithm...")
        wid = QWidget(self)
        self.setCentralWidget(wid)
        layout = QVBoxLayout()
        layout.addWidget(self.text)
        wid.setLayout(layout)


class Result(QMainWindow):
    def __init__(self, folderPath, ENVIName, ENVIFilePath):
        # Creates a grid layout
        # GUI will access the directory containing the result folders (folderPath) and
        # create the output containing the clustered image and centroid graph
        # Access ENVI file to give specific spectrum at each pixel
        QMainWindow.__init__(self)
        self.setWindowTitle("Results")
        self.setWindowIcon(QIcon('CompiledAlgorithms/DRS_Logo.jfif'))
        self.setFixedSize(1400, 700)
        self.folderPath = folderPath
        self.ENVIFilePath = ENVIFilePath
        self.ENVIName = ENVIName
        self.init_Gui()

    def init_Gui(self):
        mainLayout = QGridLayout()

        # Reads the ENVI Image for select graph function

        self.ei = Envi.EnviImage()
        self.ei.Read(self.ENVIFilePath, True, False, False)
        self.img = np.array(self.ei.Pixels)

        self.makeGraph()
        self.makePicture()
        self.makeSelectGraph()
        self.makeDescriptionGraph()
        # Image is placed top right, Graph top left, select graph bottom left and description bottom right
        mainLayout.addWidget(self.picture_label, 1, 0)
        mainLayout.addWidget(self.graphBox, 1, 1)
        mainLayout.addWidget(self.selectGraphBox)
        mainLayout.addWidget(self.descriptionBox)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)

        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(mainLayout)

    def makePicture(self):
        # Creates the clustered image from the result folder
        # Note: Image is scaled to fit the GUI

        picture = QImage(self.folderPath + self.ENVIName + "_ClusteredImage.png")
        pixmap = QPixmap(QPixmap.fromImage(picture))
        self.picture_label = QLabel()
        self.picture_label.setPixmap(pixmap)
        self.pixmap = pixmap

        # When click in the image is detected, calls function updateAbsorpGraph
        self.picture_label.mousePressEvent = self.updateAbsorpGraph
        self.picture_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.picture_label.setScaledContents(True)
        self.clustered_picture = picture


    def makeGraph(self):
        # Access the centroid absorption graph from the result folder and creates a widget for it
        graph = QLabel()
        graph.setPixmap(QPixmap(self.folderPath + self.ENVIName + "_ClusteredAbsorptionGraph.png"))
        layout = QVBoxLayout()
        layout.addWidget(graph)
        self.graphBox = QGroupBox("Absorption Graph for All Centers")
        self.graphBox.setLayout(layout)

    def makeSelectGraph(self):
        # Creates scatterplot displaying the spectrum of the currently selected pixel based on its location
        # In the ENVI Image
        self.selectGraphBox = QGroupBox("Current Selected Pixel Absorption")
        layout = QHBoxLayout()
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self.selectGraph = pg.PlotWidget()

        self.selectGraph.plot(self.ei.wavelength, self.img[0][0], pen=pg.mkPen(QColor(self.clustered_picture.pixel(0, 0)), width=3))
        layout.addWidget(self.selectGraph)
        self.selectGraphBox.setLayout(layout)

    def makeDescriptionGraph(self):
        # Information on where the files are saved and which pixdl is currently selected
        self.descriptionBox = QGroupBox("Info")
        layout = QVBoxLayout()
        self.description = QLabel("Click on the image to see the approximate absorption of each pixel.")
        self.currPixel = QLabel("Currently selected pixel: (0,0)")
        self.dirInfo = QLabel("The CSV file, Clustered Picture, and Centroid Graph are all saved to the following directory:")
        self.dirLabel = QLabel(str(self.folderPath))
        layout.addWidget(self.description)
        layout.addWidget(self.currPixel)
        layout.addWidget(self.dirInfo)
        layout.addWidget(self.dirLabel)

        self.descriptionBox.setLayout(layout)

    def updateAbsorpGraph(self, event):
        # On user click within the image, locates the location and resize the pixel location to fit the
        # ENVI image's shape
        # Color of the line corresponds to the pixel's cluster
        # Updates scatter plot and graph the spectrum of the selected pixel based on the ENVI file
        scalex = self.img.shape[1]/self.pixmap.size().width()
        scaley = self.img.shape[0]/self.pixmap.size().height()
        xOrig = event.pos().x()
        yOrig = event.pos().y()
        x = int(xOrig * scalex)
        y = int(yOrig * scaley)
        self.selectGraph.clear()
        self.selectGraph.plot(self.ei.wavelength, self.img[y][x], pen=pg.mkPen(QColor(self.clustered_picture.pixel(xOrig, yOrig)), width=3))
        self.currPixel.setText("Currently selected pixel: ({}, {})".format(str(x), str(y)))

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    GUI = MainWindow()
    # GUI.Form.show()
    sys.exit(app.exec_())
