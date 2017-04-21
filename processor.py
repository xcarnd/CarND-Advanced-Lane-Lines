# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plot
import matplotlib.image as mpimg
import numpy as np
import cv2


class Processor(object):
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
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # s plane
        image_plane = hls[:, :, 2]
        # sobel kernel size
        sobel_kernel_size = 5

        # apply sobelx operator and sobely operator
        sobelx = self.apply_sobelx(image_plane, sobel_kernel_size)
        sobely = self.apply_sobely(image_plane, sobel_kernel_size)
        # compute magnitude of gradient, absolute direction of gradient
        mag = self.get_grad_mag(sobelx, sobely)
        direct = self.get_grad_abs_dir(sobelx, sobely)

        sobelx_thresh = self.threshold(sobelx, 12, 245)
        sobely_thresh = self.threshold(sobely, 21, 228)

        mag_thresh = self.threshold(mag, 15, 190)
        dir_thresh = self.threshold(direct, 27 * np.pi / 180, 64 * np.pi / 180, False)

        mask = np.zeros_like(image_plane, dtype=np.uint8)
        mask[((sobelx_thresh == 1) & (sobely_thresh == 1)) | ((mag_thresh == 1) & (dir_thresh == 1))] = 1
        return mask

    def apply_slide_window_search(self, image):
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
        start_points_search_region_height = int(image.shape[0] * 0.5)
        mid_point = int(image.shape[1] / 2)
        left_sum = np.sum(image[image.shape[0] - start_points_search_region_height:, :mid_point], axis=0)
        left_center = np.argmax(np.convolve(window_template, left_sum)) - int(window_width / 2)
        right_sum = np.sum(image[image.shape[0] - start_points_search_region_height:, mid_point:], axis=0)
        right_center = np.argmax(np.convolve(window_template, right_sum)) - int(window_width / 2) + mid_point
        # TODO: what if failing to find left_center and right_center?
        centers.append((left_center, right_center))

        for i in range(1, num_slices):
            # calculating the y coordinates for the slice
            slice_y_max = image.shape[0] - slice_height * i
            slice_y_min = image.shape[0] - slice_height * (i + 1)
            # convolve the entire vertical slice with the window template
            conv_signal = np.convolve(window_template, np.sum(image[slice_y_min:slice_y_max, :], axis=0))
            # searching left center and right center based on the centers for the previous slice
            # searching is limited within [left_center - search_margin : left_center + search_margin]
            # in case no signal in the searching window, use center from previous slice
            l_search_min = max((left_center - search_margin, 0))
            l_search_max = min((left_center + search_margin, image.shape[1]))
            if len(conv_signal[l_search_min:l_search_max].nonzero()[0]) > 0:
                left_center = np.argmax(conv_signal[l_search_min:l_search_max]) + l_search_min - int(window_width / 2)
            # same for the right side
            r_search_min = max((right_center - search_margin, 0))
            r_search_max = min((right_center + search_margin, image.shape[1]))
            if len(conv_signal[r_search_min:r_search_max].nonzero()[0]) > 0:
                right_center = np.argmax(conv_signal[r_search_min:r_search_max]) + r_search_min - int(window_width / 2)

            centers.append((left_center, right_center))
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
        return fitting
    
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

if __name__ == '__main__':
    from camera import Camera

    processor = Processor()
    test_img = './undistorted.jpg'
    img = cv2.imread(test_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    binary = processor.extract(img)

    extracted = binary

    camera = Camera()
    camera.setup_perspective_transform(
        np.array(((597, 446),
                  (266, 670),
                  (1038, 670),
                  (682, 446)), dtype=np.float32),
        np.array(((340, 90),
                  (340, 690),
                  (940, 690),
                  (940, 90)), dtype=np.float32))

    birdview = camera.warp_perspective(extracted)
    lane_centers = processor.apply_slide_window_search(birdview)

    l_polyfit = processor.fit_polynomial_for_lane(birdview, lane_centers.T[0])
    r_polyfit = processor.fit_polynomial_for_lane(birdview, lane_centers.T[1])
    mask_img = processor.get_birdview_lane_mask_image(birdview, l_polyfit, r_polyfit)
    unwarp_mask = camera.warp_inverse_perspective(mask_img)

    result = cv2.addWeighted(img, 1, unwarp_mask, 0.3, 0)
    plot.imshow(result)
    plot.show()
