import cv2
import processing
import numpy as np
import fitting


class LaneDetectionPipeline(object):
    def __init__(self, camera, y_mpp, x_mpp):
        # camera object
        self.camera = camera
        # meters per pixel settings
        self.x_mpp = x_mpp
        self.y_mpp = y_mpp
        # fitting results for the last frames
        self.last_fit = []
        # lane points for the last frame
        self.last_fit_points = None
        # lane centers for the last frame
        self.last_lane_centers = None
        # weights when averaging with result from last frames
        self.last_fit_weights = [
            (1, ),
            (0.4, 0.6),
            (0.2, 0.2, 0.6),
            (0.1, 0.1, 0.1, 0.7)
        ]
        # how many continuous lane finding failures
        self.continuous_missing = 0

    def process_undistort(self, img):
        """For debug use. Undistort the given image only.
        """
        undistorted_img = self.camera.undistort(img)
        return undistorted_img

    def process(self, img):
        """Processing the input image, returns the processed image."""
        # undistort image
        undistorted_img = self.camera.undistort(img)

        # extract binary feature image from the undistorted image
        extracted = processing.extract(undistorted_img)

        # get warped, birdview image
        birdview = self.camera.warp_perspective(extracted)

        # if we've successfully fitting a lane in previous frames, use the latest one to find
        # the lane pixels for the current frame
        if len(self.last_fit) > 0:
            last_fit_left, last_fit_right = self.last_fit[-1]
            lp, rp, lane_centers = processing.find_lane_center_by_prior_fit(birdview, last_fit_left, last_fit_right)
        else:
            # otherwise perform a full sliding window search
            lp, rp, lane_centers = processing.find_lane_centers_by_sliding_window_search(birdview)

        if lp is None or rp is None:
            # cannot find left lane/right lane. use fitting from last frame
            # recording 1 time missing
            self.continuous_missing += 1
            # if missing for 5 continuous frames, restart sliding window searcha
            if self.continuous_missing >= 5:
                self.last_fit = []
                return undistorted_img
            else:
                # use last fitting result if any
                if len(self.last_fit) > 0:
                    # if there is any last fitting records, use it as the fitting result
                    last_fit = self.last_fit[-1]
                    last_fit_points = self.last_fit_points
                    l_polyfit, r_polyfit = last_fit
                    (lp, rp), lane_centers = last_fit_points, self.last_lane_centers
                else:
                    # no last fitting information
                    return undistorted_img
        else:
            # left lane and right lane are found. use the points to fit polynomials for the lanes.
            self.continuous_missing = 0
            l_polyfit = processing.fit_polynomial_for_lane(lp)
            r_polyfit = processing.fit_polynomial_for_lane(rp)

            self.last_fit_points = (lp, rp)
            self.last_lane_centers = lane_centers

        # averaging last 3 polylines with the current one to avoid jettering
        poly_fits = self.last_fit + [(l_polyfit, r_polyfit)]
        weights = self.last_fit_weights[len(poly_fits) - 1]
        polylines = np.array(poly_fits)
        l_polyfit = fitting.average_polylines(polylines[:, 0], weights, (0, img.shape[0]), 5)
        r_polyfit = fitting.average_polylines(polylines[:, 1], weights, (0, img.shape[0]), 5)

        # creating mask image
        mask_img = processing.get_birdview_lane_mask_image(birdview, l_polyfit, r_polyfit)
        mask_img[lp[:, 0], lp[:, 1]] = (255, 0, 0)
        mask_img[rp[:, 0], rp[:, 1]] = (0, 0, 255)

        # unwarp mask image and combined with the undistorted image
        unwarp_mask = self.camera.warp_inverse_perspective(mask_img)
        result = cv2.addWeighted(undistorted_img, 1, unwarp_mask, 0.3, 0)

        # calculate lane curvatures and off-center
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

        if self.continuous_missing == 0:
            # append the window search result when this is a successful search
            self.last_fit.append((l_polyfit, r_polyfit))
            if len(self.last_fit) > 3:
                # only store the results for last 4 frames.
                del self.last_fit[0]

        return result
