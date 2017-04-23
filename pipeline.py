import cv2
import processor
import matplotlib.pyplot as plt
import numpy as np


class LaneDetectionPipeline(object):
    def __init__(self, camera, y_mpp, x_mpp):
        self.camera = camera
        self.x_mpp = x_mpp
        self.y_mpp = y_mpp
        self.last_fit = []

    def process(self, img):
        """Processing the input image, returns the processed image."""
        undistorted_img = self.camera.undistort(img)
        extracted = processor.extract(undistorted_img)
        birdview = self.camera.warp_perspective(extracted)

        if len(self.last_fit) > 0:
            last_fit_left, last_fit_right = self.last_fit[-1]
            # do lane search based on previously fit lanes
            lane_centers = processor.find_lane_center_by_prior_fit(birdview, last_fit_left, last_fit_right)
        else:
            lane_centers = processor.find_lane_centers_by_sliding_window_search(birdview)

        l_polyfit, lp = processor.fit_polynomial_for_lane(birdview, lane_centers.T[0])
        r_polyfit, rp = processor.fit_polynomial_for_lane(birdview, lane_centers.T[1])

        mask_img = processor.get_birdview_lane_mask_image(birdview, l_polyfit, r_polyfit)

        # birdviewRgb = np.stack((birdview, birdview, birdview), axis=2)
        # birdviewRgb[lp[:, 0], lp[:, 1]] = [255, 0, 0]
        # birdviewRgb[rp[:, 0], rp[:, 1]] = [0, 0, 255]
        # birdviewRgb = cv2.addWeighted(birdviewRgb, 1, mask_img, 0.3, 0)
        #
        # plt.imshow(birdviewRgb)
        # plt.show()

        unwarp_mask = self.camera.warp_inverse_perspective(mask_img)
        result = cv2.addWeighted(img, 1, unwarp_mask, 0.3, 0)
        l_curvature = processor.compute_curvature((self.y_mpp, self.x_mpp), lp, img.shape[0])
        r_curvature = processor.compute_curvature((self.y_mpp, self.x_mpp), rp, img.shape[0])
        center_x = img.shape[1] / 2
        actual_x = np.sum(lane_centers[0]) / 2
        offset_x = actual_x - center_x
        print(offset_x)
        offset_m = self.x_mpp * offset_x
        cv2.putText(result, "Radius of Curvature: {:.1f}m".format((l_curvature + r_curvature) / 2),
                    (25, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        cv2.putText(result, "Off-center: {:.1f}m".format(offset_m),
                    (25, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        self.last_fit = [(l_polyfit, r_polyfit)]
        return result
