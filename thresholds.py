# -*- encoding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
from typing import Callable

import processor
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from camera import Camera

camera = Camera()
src_rect = np.array(((597, 446),
                     (266, 670),
                     (1038, 670),
                     (682, 446)), dtype=np.float32)
dst_rect = np.array(((400, 200),
                     (400, 720),
                     (880, 720),
                     (880, 200)), dtype=np.float32)

camera.setup_perspective_transform(src_rect, dst_rect)

class Previewer(QtCore.QObject):
    PADDING = 16

    rendered = QtCore.pyqtSignal()

    def __init__(self, qwidget, renderer: Callable[[int, int], QtGui.QImage] = None, location=(0, 0), size=(400, 300),
                 bounds=(0, 256)):
        super(Previewer, self).__init__()

        self.current_rendered_data = None
        lower_label = QLabel("Lower bound: ", qwidget)
        lower = QSlider(QtCore.Qt.Horizontal, qwidget)
        lower_indicator = QLabel("0", qwidget)

        upper = QSlider(QtCore.Qt.Horizontal, qwidget)
        upper_label = QLabel("Upper bound: ", qwidget)
        upper_indicator = QLabel("0", qwidget)

        lower.setMinimum(bounds[0])
        lower.setMaximum(bounds[1])
        lower.resize(340, 30)

        upper.setMinimum(bounds[0])
        upper.setMaximum(bounds[1])
        upper.resize(340, 30)

        lower_label.move(Previewer.PADDING + location[0], Previewer.PADDING + location[1])
        lower.move(Previewer.PADDING + 160 + location[0], Previewer.PADDING + location[1])
        lower_indicator.move(Previewer.PADDING * 2 + 160 + 340 + location[0], Previewer.PADDING + location[1])

        upper_label.move(Previewer.PADDING + location[0], Previewer.PADDING + 30 + location[1])
        upper.move(Previewer.PADDING + 160 + location[0], Previewer.PADDING + 30 + location[1])
        upper_indicator.move(Previewer.PADDING * 2 + 160 + 340 + location[0], Previewer.PADDING + 30 + location[1])

        self.scene_size = (size[0] - 2 * Previewer.PADDING, size[1] - 2 * Previewer.PADDING - 60)
        scene = QGraphicsScene()
        self.pixmap_item = QGraphicsPixmapItem()
        scene.addItem(self.pixmap_item)

        scene_widget = QGraphicsView(scene, qwidget)
        scene_widget.resize(self.scene_size[0], self.scene_size[1])
        scene_widget.move(Previewer.PADDING + location[0], Previewer.PADDING + 60 + location[1])
        self.scene_widget = scene_widget

        self.lower_indicator = lower_indicator
        self.upper_indicator = upper_indicator
        self.renderer = renderer
        self.current_bounds = list(bounds)
        self.render()

        lower.setValue(bounds[0])
        upper.setValue(bounds[1])
        lower.valueChanged.connect(lambda v: self._lower_changed(v))
        upper.valueChanged.connect(lambda v: self._upper_changed(v))

    def _lower_changed(self, new_lower):
        self.current_bounds[0] = new_lower
        self.render()

    def _upper_changed(self, new_upper):
        self.current_bounds[1] = new_upper
        self.render()

    def render(self):
        self.lower_indicator.setText(str(self.current_bounds[0]))
        self.upper_indicator.setText(str(self.current_bounds[1]))
        self.lower_indicator.adjustSize()
        self.upper_indicator.adjustSize()
        if self.renderer:
            rendered_data = self.renderer(*self.current_bounds)
            if rendered_data is not None:
                qi = None
                self.current_rendered_data = rendered_data
                if len(rendered_data.shape) == 2:
                    max_val = np.max(rendered_data)
                    if max_val > 0:
                        rendered_data = (rendered_data / max_val * 255).astype(np.uint8)
                    qi = QtGui.QImage(rendered_data.reshape(-1), rendered_data.shape[1], rendered_data.shape[0],
                                      QtGui.QImage.Format_Grayscale8)
                else:
                    qi = QtGui.QImage(rendered_data.reshape(-1), rendered_data.shape[1], rendered_data.shape[0],
                                      QtGui.QImage.Format_RGB888)
                if qi is not None:
                    content_width = self.scene_widget.contentsRect().width()
                    content_height = self.scene_widget.contentsRect().height()
                    self.pixmap_item.setPixmap(QtGui.QPixmap.fromImage(qi).scaled(content_width, content_height,
                                                                                  QtCore.Qt.KeepAspectRatio))
            self.rendered.emit()


if __name__ == '__main__':
    #test_img = './test_images/test5.jpg'
    test_img = "./undistorted.jpg"
    img = cv2.imread(test_img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gau = cv2.GaussianBlur(img_rgb, (3, 3), 0)
    gray = cv2.cvtColor(img_gau, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img_gau, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img_gau, cv2.COLOR_BGR2HSV)

    f, ((ax11, ax12, ax13), (ax21, ax22, ax23), (ax31, ax32, ax33)) = plt.subplots(nrows=3, ncols=3)
    ax11.imshow(gray, cmap='gray')
    ax21.imshow(hls[:, :, 0], cmap='gray')
    ax22.imshow(hls[:, :, 1], cmap='gray')
    ax23.imshow(hls[:, :, 2], cmap='gray')
    ax31.imshow(hsv[:, :, 0], cmap='gray')
    ax32.imshow(hsv[:, :, 1], cmap='gray')
    ax33.imshow(hsv[:, :, 2], cmap='gray')
    #plt.show()
    
    img1 = mpimg.imread('./test_images/test1.jpg')
    img2 = mpimg.imread('./test_images/test2.jpg')
    img3 = mpimg.imread('./test_images/test3.jpg')
    img4 = mpimg.imread('./test_images/test4.jpg')
    img5 = mpimg.imread('./test_images/test5.jpg')
    img6 = mpimg.imread('./test_images/test6.jpg')

    processor = Processor()
    kernel_size = 3

    def p11_renderer(low, high):
        # sobelx = processor.apply_sobelx(gray, kernel_size)
        # thresholded = processor.threshold(sobelx, low, high)
        # return thresholded
        processor.sobelx_thresh = (low, high)


    def p12_renderer(low, high):
        # return processor.threshold(hls[:, :, 2], low, high)
        processor.v_thresh = (low, high)

    def p13_renderer(low, high):
        # if p11.current_rendered_data is not None and p12.current_rendered_data is not None:
        #     result = np.zeros_like(p11.current_rendered_data, dtype=np.uint8)
        #     result[(p11.current_rendered_data > 0) | (p12.current_rendered_data > 0)] = 1
        #     birdview = camera.warp_perspective(result)
        #     return birdview
        return processor.threshold(gray, low, high, normalizing=False)


    def p21_renderer(low, high):
        # return processor.threshold(hsv[:, :, 2], low, high)
        return processor.threshold(hsv[:,:,2], low, high, normalizing=False)


    def p22_renderer(low, high):
        # return processor.threshold(processor.apply_sobelx(hsv[:, :, 2], kernel_size), low, high)
        return processor.threshold(hls[:,:,2], low, high, normalizing=False)


    def p23_renderer(low, high):
        if p21.current_rendered_data is not None and p22.current_rendered_data is not None:
            result = np.zeros_like(p21.current_rendered_data, dtype=np.uint8)
            result[(p21.current_rendered_data > 0) & (p22.current_rendered_data > 0)] = 1
            birdview = camera.warp_perspective(result)
            bv = np.stack((birdview, birdview, birdview), axis=2) * 255
            lane_centers = processor.find_lane_centers_by_sliding_window_search(birdview, debug_image=bv)
            # lf, lp = processor.fit_polynomial_for_lane(birdview, lane_centers.T[0])
            # rf, rp = processor.fit_polynomial_for_lane(birdview, lane_centers.T[1])
            # mask = processor.get_birdview_lane_mask_image(birdview, lf, rf)
            # bv = np.stack((birdview, birdview, birdview), axis=2) * 255
            # bv[lp[:, 0], lp[:, 1]] = (255, 0, 0)
            # bv[rp[:, 0], rp[:, 1]] = (255, 255, 0)
            # return cv2.addWeighted(bv, 0.5,  mask.astype(np.uint8), 0.5, 0)
            return bv
        #return processor.extract(img3)

    def p31_renderer(low, high):
        return processor.extract(img4)

    def p33_renderer(low, high):
        return processor.extract(img5)

    app = QApplication(sys.argv)

    w = QWidget()
    w.setWindowTitle("Thresholds picker")
    panel_size = (600, 480)
    w.resize(panel_size[0] * 3 + 16 * 2, panel_size[1] * 3 + 16 * 2)

    p11 = Previewer(w, renderer=p11_renderer, size=panel_size, location=(0, 0))
    p12 = Previewer(w, renderer=p12_renderer, size=panel_size, location=(panel_size[0], 0))
    p13 = Previewer(w, renderer=p13_renderer, size=panel_size, location=(panel_size[0] * 2, 0))

    p21 = Previewer(w, renderer=p21_renderer, size=panel_size, location=(0, panel_size[1]))
    p22 = Previewer(w, renderer=p22_renderer, size=panel_size, location=(panel_size[0], panel_size[1]))
                    #bounds=(0, 900))
    p23 = Previewer(w, renderer=p23_renderer, size=panel_size, location=(panel_size[0] * 2, panel_size[1]))

    p31 = Previewer(w, renderer=p31_renderer, size=panel_size, location=(0, panel_size[1] * 2))
    p33 = Previewer(w, renderer=p33_renderer, size=panel_size, location=(panel_size[0] * 2, panel_size[1] * 2))

    # p11.rendered.connect(lambda: p13.render())
    # p12.rendered.connect(lambda: p13.render())
    p21.rendered.connect(lambda: p23.render())
    p22.rendered.connect(lambda: p23.render())
    # p13.rendered.connect(lambda: p31.render())
    # p23.rendered.connect(lambda: p31.render())

    # p11.rendered.connect(lambda: p13.render() or p21.render() or p22.render() or p23.render() or p31.render() or p33.render())
    # p12.rendered.connect(lambda: p13.render() or p21.render() or p22.render() or p23.render() or p31.render() or p33.render())

    w.show()

    sys.exit(app.exec_())
