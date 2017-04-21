from processor import Processor
import numpy as np


class LaneDetectionPipeline(object):
    def __init__(self, camera):
        self.camera = camera
        self.processor = Processor()

    def process(self, img):
        """Processing the input image, returns the processed image."""
        undistorted_img = self.camera.undistort(img)
        binary = self.processor.extract(undistorted_img)
        result = np.stack((binary, binary, binary), axis=2) * 255
        return result
