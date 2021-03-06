# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plot
import cv2
import numpy as np
import os


class Camera(object):
    def __init__(self):
        self.cameraMatrix = None
        self.distortionCoeff = None
        self.persp_trans_matrix = None
        self.inv_persp_trans_matrix = None

    def calibrate(self, ncols, nrows, cal_image_paths):
        """Calibrate camera with the given chessboard images.
        """
        # generate object points array
        objp = np.mgrid[0:ncols, 0:nrows].T.reshape(-1, 2)
        objpoints = np.zeros((nrows * ncols, 3), np.float32)
        objpoints[:, :2] = objp

        objp = []
        imgp = []
        img_size = None
        for img_path in cal_image_paths:
            image = cv2.imread(img_path)
            gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if img_size is None:
                img_size = gs.shape[-1::-1]

            ret, corners = cv2.findChessboardCorners(gs, (ncols, nrows), None)
            # if corners not found, skip the specific calibration image
            if not ret:
                continue
            objp.append(objpoints)
            imgp.append(corners)
        # calibrate camera
        assert img_size is not None, "Unable to determine image size."
        assert len(objp) > 0, "No {} x {} corner pattern can be detected from the specifying calibration images."
        assert len(imgp) > 0, "No {} x {} corner pattern can be detected from the specifying calibration images."
        _, cmx, dist, _, _ = cv2.calibrateCamera(objp, imgp, img_size, None, None)
        self.cameraMatrix = cmx
        self.distortionCoeff = dist

    def undistort(self, image):
        """Undistort offered image.
        """
        assert self.cameraMatrix is not None, "Camera's not calibrated. Calibrate it first!"
        assert self.distortionCoeff is not None, "Camera's not calibrated. Calibrate it first!"
        return cv2.undistort(image, self.cameraMatrix, self.distortionCoeff)

    def setup_perspective_transform(self, src_rect, dst_rect):
        """Setup perspective transform by the specified source rect and destination rect.
        """
        self.persp_trans_matrix = cv2.getPerspectiveTransform(src_rect, dst_rect)
        self.inv_persp_trans_matrix = cv2.getPerspectiveTransform(dst_rect, src_rect)

    def warp_perspective(self, image):
        """Do perspective transform for the given image.
        """
        assert self.persp_trans_matrix is not None, "Perspective transform not setup."
        return cv2.warpPerspective(image, self.persp_trans_matrix, image.shape[1::-1], cv2.INTER_LINEAR)

    def warp_inverse_perspective(self, image):
        """Do inverse perspective transform for the image."""
        assert self.inv_persp_trans_matrix is not None, "Perspective transform not setup."
        return cv2.warpPerspective(image, self.inv_persp_trans_matrix, image.shape[1::-1], cv2.INTER_LINEAR)

if __name__ == '__main__':
    cal_images_dir = "./camera_cal"
    camera = Camera()
    camera.calibrate(9, 6, [cal_images_dir + "/" + p for p in os.listdir(cal_images_dir)])

    test_img = "./camera_cal/calibration1.jpg"
    img = cv2.imread(test_img)
    undistorted = camera.undistort(img)

    f, (ax1, ax2) = plot.subplots(1, 2, squeeze=True, figsize=(8, 3))
    ax1.imshow(img)
    ax1.set_title('Original image')
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted image')
    f.savefig('./output_images/camera_calibration.png')

    # setup perspective transform
    src_rect = np.array(((595, 447),
                         (237, 697),
                         (1085, 697),
                         (686, 447)), dtype=np.float32)
    # src rect for challenge video
    src_rect = np.array(((607, 472),
                         (298, 707),
                         (1123, 707),
                         (710, 472)), dtype=np.float32)

    dst_rect = np.array(((300, 0),
                         (300, 720),
                         (980, 720),
                         (980, 0)), dtype=np.float32)

    camera.setup_perspective_transform(src_rect, dst_rect)

    next_img = "./debug/frame0000.jpg"
    img = cv2.imread(next_img)
    undistorted2 = camera.undistort(img)
    birdview = camera.warp_perspective(undistorted2)
    plot.imshow(birdview)
    plot.show()
    cv2.imwrite("./undistorted.jpg", undistorted2)
