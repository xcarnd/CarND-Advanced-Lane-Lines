from processor import Processor
import cv2
import matplotlib.pyplot as plt


class LaneDetectionPipeline(object):
    def __init__(self, camera):
        self.camera = camera
        self.processor = Processor()

    def process(self, img):
        """Processing the input image, returns the processed image."""
        undistorted_img = self.camera.undistort(img)
        extracted = self.processor.extract(undistorted_img)
        birdview = self.camera.warp_perspective(extracted)

        lane_centers = self.processor.apply_slide_window_search(birdview)
        plt.imshow(birdview)
        plt.show()

        l_polyfit,_ = self.processor.fit_polynomial_for_lane(birdview, lane_centers.T[0])
        r_polyfit,_ = self.processor.fit_polynomial_for_lane(birdview, lane_centers.T[1])

        mask_img = self.processor.get_birdview_lane_mask_image(birdview, l_polyfit, r_polyfit)

        unwarp_mask = self.camera.warp_inverse_perspective(mask_img)
        result = cv2.addWeighted(img, 1, unwarp_mask, 0.3, 0)
        return result
