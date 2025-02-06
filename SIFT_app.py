#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import numpy as np
import sys

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture("/home/fizzer/SIFT_app/Leopard.mp4")
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)

    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                 bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        # Read the current frame from the camera/video
        ret, frame = self._camera_device.read()
        if not ret:
            print("Failed to read frame from camera.")
            return

        # Check if template_path exists before trying to load the template image
        if not hasattr(self, "template_path") or self.template_path is None:
            print("No template selected. Displaying camera feed only.")
        else:
            img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)  
            if img is None:
                print(f"Failed to load template image: {self.template_path}")
            else:
                # SIFT matching only if the template image is valid
                sift = cv2.SIFT_create()
                kp_image, desc_image = sift.detectAndCompute(img, None)
                grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

                if desc_image is not None and desc_grayframe is not None:
                    # FLANN matcher
                    index_params = dict(algorithm=0, trees=5)
                    search_params = dict()
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

                    if matches:
                        good_points = [m for m, n in matches if m.distance < 0.6 * n.distance]
                        if len(good_points) >= 4:
                            # Compute homography
                            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
                            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

                            if matrix is not None:
                                h, w = img.shape
                                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                                dst = cv2.perspectiveTransform(pts, matrix)
                                cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
                            else:
                                print("Homography computation failed.")
                        else:
                            print(f"Not enough matches found: {len(good_points)} / 4")
                    else:
                        print("No matches found.")

        # Always update the QLabel with the current frame
        pixmap = self.convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap)




    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())
