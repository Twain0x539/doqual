from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog, QLabel, QHBoxLayout, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap, QFont
from processing import *
import cv2


# Подкласс QMainWindow для настройки главного окна приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.W = 1280
        self.H = 720
        self.setFixedSize(QSize(self.W, self.H))
        self.image_processor = ImageProcessor()
        self.setWindowTitle("Оценщик качества")

        self.start_layout = QVBoxLayout()

        self.label_1 = QLabel('Фото на документы должно :\n' + \
                              '1) Иметь однородный задний фон\n' + \
                              '2) Содержать только одно лицо\n' + \
                              '3) Лицо должно выражать нейтральную эмоцию\n' + \
                              '4) Глаза должны быть открыты, лицо не перекрыто волосами\n' + \
                              '5) Освещение равномерное, отсутствует тень\n',
                              self)
        self.label_1.setFont(QFont("Arial", 15))
        self.label_1.move(100, 100)
        self.label_1.setStyleSheet("border: 1px solid black;")

        self.load_image_button = QPushButton("Загрузить изображение")
        self.load_image_button.setCheckable(True)
        self.load_image_button.clicked.connect(self.load_image)

        self.start_layout.addWidget(self.label_1)
        self.start_layout.addWidget(self.load_image_button)

        self.start_widget = QWidget(self)
        self.start_widget.setLayout(self.start_layout)
        tip_w = 700
        tip_h = 600
        self.start_widget.setGeometry((self.W - tip_w) // 2, (self.H - tip_h) // 2, tip_w, tip_h)

        self.start_widget.show()
        self.font = QFont("Arial", 10)

    def load_image(self):

        dialog = QFileDialog(self)
        filename = dialog.getOpenFileName(self, "Open Image", filter="Images (*.png *.xpm *.jpg)")[0]
        if filename != "":
            self.start_widget.hide()
            img = cv2.imread(filename)
            processing_result = self.image_processor.process(img)
            boxes, tgt_id, ver_lst, face_qual, bgr_unif, no_bgr_img = processing_result

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
            lbl_quality.setFont(self.font)

            if face_qual > 50:
                lbl_quality.setText(f'<span style="color:black;"> Качество лица: {face_qual}</span> <br>'
                                           f'<span style="color:green;">Высокое качество лица</span>')
            elif bgr_unif < 50:
                lbl_quality.setText(f'<span style="color:black;"> Качество лица: {face_qual}</span> <br>'
                                           f'<span style="color:red;">Низкое качество лица</span>')


            lbl_bgr_score = QLabel(self)
            if bgr_unif > 50:
                lbl_bgr_score.setText(f'<span style="color:black;"> Однородность заднего фона: {bgr_unif}</span> <br>'
                                           f'<span style="color:green;">Высокая однородность заднего фона</span>')
            elif bgr_unif < 50:
                lbl_bgr_score.setText(f'<span style="color:black;"> Однородность заднего фона: {bgr_unif}</span> <br>'
                                           f'<span style="color:red;">Низкая однородность заднего фона</span>')


            lbl_multiple_faces = QLabel(self)
            if len(boxes) > 1:
                lbl_multiple_faces.setText(f'<span style="color:black;">На фотографии {len(boxes)} лиц</span> <br>'
                                           f'<span style="color:red;">На фотографии более 1 лица!</span>')
            elif len(boxes) == 0:
                lbl_multiple_faces.setText(f'<span style="color:black;">На фотографии {len(boxes)} лиц</span> \n'
                                           f'<span style="color:red;">На фотографии не найдено лиц!</span>')

            lbl_tip = QLabel(self)
            if len(boxes) > 1 or  len(boxes) == 0 or bgr_unif < 50 or face_qual < 45:
                lbl_tip.setText(f'<span style="color:red;">Использование фотографию на документе не рекомендуется!</span>')
            elif len(boxes) == 0:
                lbl_tip.setText(f'<span style="color:red;">Фотография подходит для использования на документе</span>')



            image_desc_layout = QVBoxLayout(self)
            buttons_widget = QWidget(self)
            buttons_layout = QHBoxLayout(self)
            return_button = QPushButton("Изменить изображение")
            return_button.setCheckable(True)
            return_button.clicked.connect(self.load_image)
            continue_button = QPushButton("Продолжить")
            continue_button.setCheckable(True)
            #continue_button.clicked.connect()
            buttons_layout.addWidget(return_button)
            buttons_layout.addWidget(continue_button)
            buttons_widget.setLayout(buttons_layout)
            image_desc_layout.addWidget(lbl_quality)
            image_desc_layout.addWidget(lbl_bgr_score)
            image_desc_layout.addWidget(lbl_multiple_faces)
            image_desc_layout.addWidget(lbl_tip)
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


    def process_image(self, image, kps):
        doc_images = self.image_processor.estimate_doc_format(image, kps, remove_bg=False)

        images_layout = QHBoxLayout(self)

        image_labels = []
        for i in range(len(doc_images)):
            image_labels.append()
