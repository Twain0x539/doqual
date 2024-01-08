from PyQt6.QtWidgets import QApplication
from main_window import MainWindow

import sys

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()