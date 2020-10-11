import GUI
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    GUI = GUI.MainWindow()
    sys.exit(app.exec_())