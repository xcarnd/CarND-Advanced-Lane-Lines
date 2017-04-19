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

    def extract(self, image, sobel_kernel_size=3):
        """Extract lane line from the specifying img
        """
        # convert to grayscale
        gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # histogram equalization
        gs = cv2.equalizeHist(gs)

        # apply sobelx operator and sobely operator
        sobelx = self.apply_sobelx(gs, sobel_kernel_size)
        sobely = self.apply_sobely(gs, sobel_kernel_size)
        # compute magnitude of gradient, absolute direction of gradient
        mag = self.get_grad_mag(sobelx, sobely)
        direct = self._grad_abs_dir(sobelx, sobely)

        return gs


if __name__ == '__main__':
    pipeline = Processor()
    test_img = './test_images/straight_lines1.jpg'
    img = cv2.imread(test_img)
    img2 = pipeline.extract(img)
    img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img32 = cv2.equalizeHist(img3)
    f, (ax1, ax2) = plot.subplots(nrows=1, ncols=2)
    ax1.imshow(img32, cmap='gray')
    ax2.imshow(img2, cmap='gray')
    plot.show()
