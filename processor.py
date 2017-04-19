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
    def get_grad_mag(sobelx = None, sobely = None):
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
        # convert to grayscale
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
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

if __name__ == '__main__':
    pipeline = Processor()
    test_img = './test_images/test6.jpg'
    img = cv2.imread(test_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    binary = pipeline.extract(img)

    f, (ax1, ax2) = plot.subplots(nrows=1, ncols=2)
    ax1.imshow(img)
    ax2.imshow(np.stack((binary, binary, binary), axis=2) * 255)
    plot.show()
