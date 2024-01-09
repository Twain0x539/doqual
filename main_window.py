from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog, QLabel, QHBoxLayout, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap, QFont
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
        self.font = QFont("Arial", 10)

    def load_image(self):
        dialog = QFileDialog(self)
        filename = dialog.getOpenFileName(self, "Open Image", filter="Images (*.png *.xpm *.jpg)")[0]

        if filename != "":
            self.load_image_button.hide()
            img = cv2.imread(filename)
            processing_result = self.image_processor.process(img)
            boxes, tgt_id, ver_lst, face_qual, bgr_unif = processing_result

            for i in range(len(boxes)):
                tgt_box = boxes[i]
                if i == tgt_id:
                    color = (0, 0, 255)
                else:
                    color = (0, 0, 255 / 4)
                start_point = (int(tgt_box[0]), int(tgt_box[1]))
                end_point = (int(tgt_box[2]), int(tgt_box[3]))
                img = cv2.rectangle(img, start_point, end_point, color, 2)

            tgt_ver_lst = ver_lst[tgt_id]
            for pt in tgt_ver_lst.T:
                img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius=0, color=(255, 0, 0), thickness=3)

            pixmap = QPixmap()
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            q_img = QImage(img.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
            pixmap.convertFromImage(q_img)

            lbl_quality = QLabel(self)
            lbl_quality.setText(f"Качество лица: {face_qual}")
            lbl_quality.setStyleSheet("QLabel {color : blue;}")
            lbl_quality.setFont(self.font)


            lbl_bgr_score = QLabel(self)
            if bgr_unif > 50:
                lbl_bgr_score.setText(f'<span style="color:black;"> Однородность заднего фона: {bgr_unif}</span> <br>'
                                           f'<span style="color:green;">Однородность заднего фона не ниже 50%</span>')
            elif bgr_unif < 50:
                lbl_bgr_score.setText(f'<span style="color:black;"> Однородность заднего фона: {bgr_unif}</span> <br>'
                                           f'<span style="color:red;">Однородность заднего фона ниже 50%</span>')


            lbl_multiple_faces = QLabel(self)
            if len(boxes) > 1:
                lbl_multiple_faces.setText(f'<span style="color:black;">На фотографии {len(boxes)} лиц</span> <br>'
                                           f'<span style="color:red;">Внимание! На фотографии более 1 лица!</span>')
            elif len(boxes) == 0:
                lbl_multiple_faces.setText(f'<span style="color:black;">На фотографии {len(boxes)} лиц</span> \n'
                                           f'<span style="color:red;">Внимание! На фотографии не найдено лиц!</span>')



            image_desc_layout = QVBoxLayout(self)
            buttons_widget = QWidget(self)
            buttons_layout = QHBoxLayout(self)
            return_button = QPushButton("Вернуться")
            use_another_button = QPushButton("Изменить зображение")
            buttons_layout.addWidget(return_button)
            buttons_layout.addWidget(use_another_button)
            buttons_widget.setLayout(buttons_layout)
            image_desc_layout.addWidget(lbl_quality)
            image_desc_layout.addWidget(lbl_bgr_score)
            image_desc_layout.addWidget(lbl_multiple_faces)
            image_desc_layout.addWidget(buttons_widget)
            image_desc_widget = QWidget(self)
            image_desc_widget.setLayout(image_desc_layout)
            image_label = QLabel(self)
            image_label.setPixmap(pixmap)


            processing_result_widget = QWidget(self)
            processing_result_layout = QHBoxLayout(self)
            processing_result_layout.addWidget(image_label)
            processing_result_layout.addWidget(image_desc_widget)
            processing_result_widget.setLayout(processing_result_layout)

            self.setCentralWidget(processing_result_widget)
            self.show()