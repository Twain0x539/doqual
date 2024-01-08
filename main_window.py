from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog, QLabel, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from processing import *
import cv2


# Подкласс QMainWindow для настройки главного окна приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_processor = ImageProcessor()
        self.setWindowTitle("Оценщик качества")
        self.load_image_button = QPushButton("Загрузить изображение")
        self.load_image_button.setCheckable(True)
        self.load_image_button.clicked.connect(self.load_image)
        self.setCentralWidget(self.load_image_button)
        self.setFixedSize(QSize(1280, 720))

    def load_image(self):
        dialog = QFileDialog(self)
        filename = dialog.getOpenFileName(self, "Open Image", filter="Images (*.png *.xpm *.jpg)")[0]

        if filename != "":
            self.load_image_button.hide()
            img = cv2.imread(filename)
            processing_result = self.image_processor.process(img)
            boxes, tgt_id, ver_lst, face_qual, bgr_unif = processing_result

            tgt_ver_lst = ver_lst[tgt_id]
            tgt_box = boxes[tgt_id]

            start_point = (int(tgt_box[0]), int(tgt_box[1]))
            end_point = (int(tgt_box[2]), int(tgt_box[3]))

            img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)

            for pt in tgt_ver_lst.T:
                print(pt)
                img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius=0, color=(255, 0, 0), thickness=3)

            pixmap = QPixmap()
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            q_img = QImage(img.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
            pixmap.convertFromImage(q_img)

            # lbl = QLabel(self)
            # lbl.setPixmap(pixmap)
            # lbl.move(250,80)
            # lbl.show()
            #
            # hbox = QHBoxLayout(self)
            # hbox.addWidget(lbl)
            #
            # self.setLayout(hbox)
            # self.show()

            lbl = QLabel(self)
            lbl.setPixmap(pixmap)
            hbox = QHBoxLayout(self)
            hbox.addWidget(lbl)
            self.setCentralWidget(lbl)
            self.setLayout(hbox)
            self.move(300, 200)
            self.setWindowTitle('image')
            self.show()