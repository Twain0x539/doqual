

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog
from processing import *


# Подкласс QMainWindow для настройки главного окна приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Оценщик качества")
        button = QPushButton("Загрузить изображение")
        button.setCheckable(True)
        button.clicked.connect(self.load_image)
        self.setCentralWidget(button)
        self.setFixedSize(QSize(1280, 720))

    def load_image(self):
        dialog = QFileDialog(self)
        filename = dialog.getOpenFileName(self, "Open Image", filter="Images (*.png *.xpm *.jpg)")
        print(filename)[0]
        if filename != "":
            print("Loading image")

