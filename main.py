import sys

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

import os

directory = os.path.dirname(os.path.abspath(__file__)) + "\\data\\"
person = False


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)

    def change_cam(self):
        self.camera = cv2.VideoCapture(0)

    def change_vid(self):
        self.camera = cv2.VideoCapture("C:\\Users\\nisha\\PycharmProjects\\opencv3\\data\\sample.mp4")


class PedDetectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage()
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        text_file = directory + "MobileNetSSD_deploy.prototxt.txt"
        caffe_file = directory + "MobileNetSSD_deploy.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(text_file, caffe_file)

    def image_data_slot(self, image_data):
        global person
        (h, w) = image_data.shape[:2]
        person = False
        blob = cv2.dnn.blobFromImage(cv2.resize(image_data, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Only display when there's a person
                if self.CLASSES[idx] == "person":
                    label = "{}: {:.2f}%".format("Detected", confidence * 100)
                    person = True
                    cv2.rectangle(image_data, (startX, startY), (endX, endY),
                                  self.COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(image_data, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

        main_widget.label1.setText("Detected: " + str(person))

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage

        image = q_image(image.data,
                        width,
                        height,
                        bytes_per_line,
                        q_image.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ped_detection_widget = PedDetectionWidget()

        self.record_video = RecordVideo()

        image_data_slot = self.ped_detection_widget.image_data_slot

        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.ped_detection_widget)
        self.run_button = QtWidgets.QPushButton("Start")
        layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.record_video.start_recording)

        self.run_button2 = QtWidgets.QPushButton("Cam Source")
        layout.addWidget(self.run_button2)
        self.run_button2.clicked.connect(self.record_video.change_cam)

        self.run_button3 = QtWidgets.QPushButton("Video Source")
        layout.addWidget(self.run_button3)
        self.run_button3.clicked.connect(self.record_video.change_vid)

        self.label1 = QtWidgets.QLabel("Hello")
        layout.addWidget(self.label1)
        self.label1.setText("Detected: " + str(person))

        newfont = QtGui.QFont("Times", 12, QtGui.QFont.Bold)
        self.label1.setFont(newfont)

        self.setLayout(layout)


app = QtWidgets.QApplication(sys.argv)
main_window = QtWidgets.QMainWindow()
main_widget = MainWidget()
main_window.setCentralWidget(main_widget)
main_window.show()
sys.exit(app.exec_())
