# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plot
import matplotlib.image as mpimg
import numpy as np
import cv2


class Processor(object):
    def __init__(self):
        self.sobelx_thresh = (40, 170)
        self.v_thresh = (200, 256)
        self.s_thresh = (100, 256)

    @staticmethod
    def apply_sobelx(gs, kernel_size):
        """Apply Sobel x operator to the provided grayscale image.
        """
        return cv2.Sobel(gs, cv2.CV_64F, 1, 0, ksize=kernel_size)

    @staticmethod
    def apply_sobely(gs, kernel_size):
        """Apply Sobel y operator tor the provided grayscale image.
        """
        return cv2.Sobel(gs, cv2.CV_64F, 0, 1, ksize=kernel_size)

    @staticmethod
    def get_grad_mag(sobelx=None, sobely=None):
        """Get the magnitude of gradient by the sobelx values and sobely values.
        
        sobelx values and sobely values must be got from the same kernel size
        """
        return np.sqrt(sobelx ** 2 + sobely ** 2)

    @staticmethod
    def get_grad_abs_dir(sobelx, sobely):
        """Get the absolute value of gradient direction (in radian) by the sobelx values and sobely values.
        
        sobelx values and sobely values must be got from the same kernel size
        """
        return np.arctan2(np.abs(sobely), np.abs(sobelx))

    @staticmethod
    def threshold(img_plane, lower_bound, upper_bound, normalizing=True):
        """Create mask by thresholding (between [lower_bound, upper_bound) ) the specify image plane data. 
        
        By setting normalizing to True, the image plane data will first be normalized into the range [0, 255] before 
        thresholding.
        
        Returns a mask image plane with points within thresholds set to 1 and 0 otherwise."""
        img_data = img_plane
        if normalizing:
            img_data = np.uint8(img_data / np.max(img_data) * 255)
        mask = np.zeros_like(img_plane, dtype=np.uint8)
        mask[(img_data >= lower_bound) & (img_data < upper_bound)] = 1
        return mask

    def extract(self, image):
        """Extract lane line from the specifying img
        """
        # convert to hls
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        hls = cv2.cvtColor(blurred, cv2.COLOR_RGB2HLS)

        # v plane
        v_plane = hsv[:, :, 2]
        s_plane = hls[:, :, 2]
        # sobel kernel size
        sobel_kernel_size = 3

        # masking applied to s plane
        # apply sobelx operator and sobely operator
        v_sobelx = np.abs(self.apply_sobelx(v_plane, sobel_kernel_size))
        v_sobely = np.abs(self.apply_sobely(v_plane, sobel_kernel_size))
        v_sobelx_thresh = self.threshold(v_sobelx, self.sobelx_thresh[0], self.sobelx_thresh[1])
        v_thresh = self.threshold(v_plane, self.v_thresh[0], self.v_thresh[1], normalizing=False)
        s_thresh = self.threshold(s_plane, self.s_thresh[0], self.s_thresh[1], normalizing=False)

        mask = np.zeros_like(v_plane, dtype=np.uint8)
        mask[(v_sobelx_thresh == 1) | ((s_thresh == 1) & (v_thresh == 1))] = 1

        return mask

    def draw_polygon(self, image, poly_points, color):
        cv2.polylines(image, np.int32([poly_points]), 1, color, 3)

    def find_lane_centers_by_sliding_window_search(self, image, debug_image=None):
        """Find left lane and right lane by applying slide window search al
        """
        centers = []

        num_slices = 10
        # divide the whole images into 10 horizontal strips. calculate the height for each strip
        slice_height = int(image.shape[0] / num_slices)

        # window template, which will be convolved with the slice the find the peak of signal
        window_width = 50
        window_template = np.ones(window_width)

        search_margin = 100

        # determine the starting point for searching. summing the quarter bottom of image to get slice
        start_points_search_region_height = int(image.shape[0] * 0.25)
        mid_point = int(image.shape[1] / 2)
        left_sum = np.sum(image[image.shape[0] - start_points_search_region_height:, :mid_point], axis=0)
        left_center = np.argmax(np.convolve(window_template, left_sum)) - int(window_width / 2)
        right_sum = np.sum(image[image.shape[0] - start_points_search_region_height:, mid_point:], axis=0)
        right_center = np.argmax(np.convolve(window_template, right_sum)) - int(window_width / 2) + mid_point
        # TODO: what if failing to find left_center and right_center?
        centers.append((left_center, right_center))

        if debug_image is not None:
            cv2.rectangle(debug_image,
                          (int(left_center - window_width / 2), image.shape[0]),
                          (int(left_center + window_width / 2), image.shape[0] - slice_height),
                          (0, 255, 0), 3)
            cv2.rectangle(debug_image,
                          (int(right_center - window_width / 2), image.shape[0]),
                          (int(right_center + window_width / 2), image.shape[0] - slice_height),
                          (0, 255, 0), 3)

        for i in range(1, num_slices):
            # calculating the y coordinates for the slice
            slice_y_max = image.shape[0] - slice_height * i
            slice_y_min = image.shape[0] - slice_height * (i + 1)
            # convolve the entire vertical slice with the window template
            conv_signal = np.convolve(window_template, np.sum(image[slice_y_min:slice_y_max, :], axis=0))
            # searching left center and right center based on the centers for the previous slice
            # searching is limited within [left_center - search_margin : left_center + search_margin]
            # in case no signal in the searching window, use center from previous slice
            offset = window_width / 2

            l_search_min = int(max((left_center - search_margin + offset, 0)))
            l_search_max = int(min((left_center + search_margin + offset, image.shape[1])))
            if len(conv_signal[l_search_min:l_search_max].nonzero()[0]) > 0:
                argmax = np.argmax(conv_signal[l_search_min:l_search_max])
                inverse_argmax = int((l_search_max - l_search_min) -
                                     np.argmax(conv_signal[l_search_max:l_search_min:-1]))
                l_center = int((argmax + inverse_argmax) / 2 + l_search_min - offset)
                left_center = l_center
            if debug_image is not None:
                cv2.circle(debug_image, (left_center, int((slice_y_min + slice_y_max) / 2)), 1, (255, 0, 0), 2)
                cv2.rectangle(debug_image,
                              (int(left_center - window_width / 2), slice_y_min),
                              (int(left_center + window_width / 2), slice_y_max),
                              (0, 255, 0), 3)

            # same for the right side
            r_search_min = int(max((right_center - search_margin + offset, 0)))
            r_search_max = int(min((right_center + search_margin + offset, image.shape[1])))
            if len(conv_signal[r_search_min:r_search_max].nonzero()[0]) > 0:
                argmax = np.argmax(conv_signal[r_search_min:r_search_max])
                inverse_argmax = int((r_search_max - r_search_min) -
                                     np.argmax(conv_signal[r_search_max:r_search_min:-1]))
                r_center = int((argmax + inverse_argmax) / 2 + r_search_min - offset)
                right_center = r_center
            if debug_image is not None:
                cv2.circle(debug_image, (right_center, int((slice_y_min + slice_y_max) / 2)), 1, (255, 0, 0), 2)
                cv2.rectangle(debug_image,
                              (int(right_center - window_width / 2), slice_y_min),
                              (int(right_center + window_width / 2), slice_y_max),
                              (0, 255, 0), 3)

            centers.append((left_center, right_center))
        return np.array(centers)

    def find_lane_center_by_prior_fit(self, image, left_fit_params, right_fit_params, num_windows=10):
        """Perform a lanes search by using previous fit parameters.
        """
        # divide the whole images into 10 horizontal strips. calculate the height for each strip
        slice_height = int(image.shape[0] / num_windows)

        l_a, l_b, l_c = left_fit_params
        r_a, r_b, r_c = right_fit_params

        centers = []
        for i in range(num_windows):
            # calculating the y coordinates for the slice
            slice_y_max = image.shape[0] - slice_height * i
            slice_y_min = image.shape[0] - slice_height * (i + 1)
            # calculating the window center based the the provided fit parameters
            center_y = (slice_y_max + slice_y_min) / 2
            l_center_x = l_a * center_y ** 2 + l_b * center_y + l_c
            r_center_x = r_a * center_y ** 2 + r_b * center_y + r_c
            centers.append((l_center_x, r_center_x))
        return np.array(centers)

    def fit_polynomial_for_lane(self, image, centers):
        num_slices = 10
        # divide the whole images into 10 horizontal strips. calculate the height for each strip
        slice_height = int(image.shape[0] / num_slices)

        # window template, which will be convolved with the slice the find the peak of signal
        window_width = 50

        window_points = []
        for i in range(num_slices):
            # calculating the y coordinates for the slice
            slice_y_max = image.shape[0] - slice_height * i
            slice_y_min = image.shape[0] - slice_height * (i + 1)
            window_min = max((0, int(centers[i] - window_width / 2)))
            window_max = min((image.shape[1], int(centers[i] + window_width / 2)))
            window = image[slice_y_min:slice_y_max, window_min:window_max]
            y_coords, x_coords = window.nonzero()
            points = np.stack((y_coords + slice_y_min, x_coords + window_min), axis=1)
            window_points.append(points)
        window_points = np.concatenate(window_points)
        fitting = np.polyfit(window_points[:, 0], window_points[:, 1], 2)
        return fitting, window_points

    def get_birdview_lane_mask_image(self, image, lane_left_fit, lane_right_fit, color=(0, 255, 0)):
        l_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        l_x = lane_left_fit[0] * l_y ** 2 + lane_left_fit[1] * l_y + lane_left_fit[2]
        l_x = l_x.astype(np.int)
        l_y = l_y.astype(np.int)

        r_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        r_x = lane_right_fit[0] * r_y ** 2 + lane_right_fit[1] * r_y + lane_right_fit[2]
        r_x = r_x.astype(np.int)
        r_y = r_y.astype(np.int)

        left_points = np.stack((l_x, l_y), axis=1)
        right_points = np.stack((r_x, r_y), axis=1)
        poly_points = np.concatenate((left_points[-1::-1, :], right_points))

        mask = np.zeros_like(np.stack((image, image, image), axis=2), dtype=np.uint8)
        return cv2.fillPoly(mask, np.int32([poly_points]), color=color)

    def compute_curvature(self, mpp, lane_points, at_y):
        a, b, c = np.polyfit(lane_points[:, 0] * mpp[0], lane_points[:, 1] * mpp[1], 2)
        print(a, b, c)

        d1 = 2 * a * at_y * mpp[0] + b
        d2 = 2 * a
        curvature = ((1 + d1 ** 2) ** (3 / 2)) / abs(d2)
        return curvature


if __name__ == '__main__':
    from camera import Camera

    processor = Processor()
    test_img = './undistorted.jpg'
    img = cv2.imread(test_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    binary = processor.extract(img)

    extracted = binary

    camera = Camera()
    src_rect = np.array(((597, 446),
                         (266, 670),
                         (1038, 670),
                         (682, 446)), dtype=np.float32)
    dst_rect = np.array(((300, 0),
                         (300, 720),
                         (980, 720),
                         (980, 0)), dtype=np.float32)

    camera.setup_perspective_transform(src_rect, dst_rect)

    # processor.draw_polygon(extracted, src_rect, 1)
    plot.imshow(extracted)
    plot.show()
    birdview = camera.warp_perspective(extracted)
    bvrgb = np.stack((birdview, birdview, birdview), axis=2) * 255
    overlapped = cv2.addWeighted(bvrgb, 0.5, camera.warp_perspective(img), 0.5, 0)
    plot.imshow(overlapped)
    plot.show()

    lane_centers = processor.find_lane_centers_by_sliding_window_search(birdview)

    l_polyfit, lp = processor.fit_polynomial_for_lane(birdview, lane_centers.T[0])
    r_polyfit, rp = processor.fit_polynomial_for_lane(birdview, lane_centers.T[1])
    mask_img = processor.get_birdview_lane_mask_image(birdview, l_polyfit, r_polyfit)
    mask_img[lp[:, 0], lp[:, 1]] = [255, 0, 0]
    mask_img[rp[:, 0], rp[:, 1]] = [0, 0, 255]

    plot.imshow(mask_img)
    plot.show()
    unwarp_mask = camera.warp_inverse_perspective(mask_img)

    result = cv2.addWeighted(img, 1, unwarp_mask, 0.3, 0)

    print(processor.compute_curvature((30 / 720, 3.7 / 600), lp))
    print(processor.compute_curvature((30 / 720, 3.7 / 600), rp))
