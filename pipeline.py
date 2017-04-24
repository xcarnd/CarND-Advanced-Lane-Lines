import cv2
import processing
import matplotlib.pyplot as plt
import numpy as np
import fitting


class LaneDetectionPipeline(object):
    def __init__(self, camera, y_mpp, x_mpp):
        self.camera = camera
        self.x_mpp = x_mpp
        self.y_mpp = y_mpp
        self.last_fit = []
        self.last_fit_weights = [
            (1, ),
            (0.43, 0.57),
            (0.22, 0.33, 0.45),
            (0.1, 0.2, 0.3, 0.4)
        ]

    def process(self, img):
        """Processing the input image, returns the processed image."""
        undistorted_img = self.camera.undistort(img)
        extracted = processing.extract(undistorted_img)
        birdview = self.camera.warp_perspective(extracted)

        if len(self.last_fit) > 0:
            last_fit_left, last_fit_right = self.last_fit[-1]
            # do lane search based on previously fit lanes
            lane_centers = processing.find_lane_center_by_prior_fit(birdview, last_fit_left, last_fit_right)
        else:
            lane_centers = processing.find_lane_centers_by_sliding_window_search(birdview)

        l_polyfit, lp = processing.fit_polynomial_for_lane(birdview, lane_centers.T[0])
        r_polyfit, rp = processing.fit_polynomial_for_lane(birdview, lane_centers.T[1])

        poly_fits = self.last_fit + [(l_polyfit, r_polyfit)]

        weights = self.last_fit_weights[len(poly_fits) - 1]
        polylines = np.array(poly_fits)
        l_polyfit = fitting.average_polylines(polylines[:, 0], weights, (0, img.shape[0]), 5)
        r_polyfit = fitting.average_polylines(polylines[:, 1], weights, (0, img.shape[0]), 5)

        mask_img = processing.get_birdview_lane_mask_image(birdview, l_polyfit, r_polyfit)
        mask_img[lp[:, 0], lp[:, 1]] = (255, 0, 0)
        mask_img[rp[:, 0], rp[:, 1]] = (0, 0, 255)

        unwarp_mask = self.camera.warp_inverse_perspective(mask_img)
        result = cv2.addWeighted(undistorted_img, 1, unwarp_mask, 0.3, 0)
        l_curvature = processing.compute_curvature((self.y_mpp, self.x_mpp), lp, img.shape[0])
        r_curvature = processing.compute_curvature((self.y_mpp, self.x_mpp), rp, img.shape[0])
        center_x = img.shape[1] / 2
        actual_x = np.sum(lane_centers[0]) / 2
        offset_x = actual_x - center_x
        offset_m = self.x_mpp * offset_x
        cv2.putText(result, "Radius of Curvature: {:.1f}m".format((l_curvature + r_curvature) / 2),
                    (25, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        cv2.putText(result, "Off-center: {:.1f}m".format(offset_m),
                    (25, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        self.last_fit.append((l_polyfit, r_polyfit))
        if len(self.last_fit) > 3:
            del self.last_fit[0]

        return result
