from camera import Camera
from moviepy.editor import VideoFileClip
from pipeline import LaneDetectionPipeline
import numpy as np

import os

# camera calibration
cal_images_dir = "./camera_cal"
camera = Camera()
camera.calibrate(9, 6, [cal_images_dir + "/" + p for p in os.listdir(cal_images_dir)])

# setup perspective transform
camera.setup_perspective_transform(
    np.array(((597, 446),
              (266, 670),
              (1038, 670),
              (682, 446)), dtype=np.float32),
    np.array(((340, 90),
              (340, 690),
              (940, 690),
              (940, 90)), dtype=np.float32))

pipeline = LaneDetectionPipeline(camera)

# clip_output_path = "./output.mp4"
# input_clip = VideoFileClip('./challenge_video.mp4')
# output_clip = input_clip.fl_image(lambda img: pipeline.process(img))
# output_clip.write_videofile(clip_output_path, audio=False)

import cv2
import matplotlib.pyplot as plt

output_path = "./test2.jpg"
input_img = './test_images/test1.jpg'
img = cv2.imread(input_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
output = pipeline.process(img)
cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
plt.imshow(output)
plt.show()
