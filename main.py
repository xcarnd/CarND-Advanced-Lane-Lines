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
src_rect = np.array(((595, 447),
                     (237, 697),
                     (1085, 697),
                     (686, 447)), dtype=np.float32)
# src rect for challenge video
# src_rect = np.array(((607, 472),
#                      (298, 707),
#                      (1123, 707),
#                      (710, 472)), dtype=np.float32)
dst_rect = np.array(((300, 0),
                     (300, 720),
                     (980, 720),
                     (980, 0)), dtype=np.float32)

camera.setup_perspective_transform(src_rect, dst_rect)

# meters per pixel settings
y_mpp = 30 / 720
x_mpp = 3.7 / (980-300)

pipeline = LaneDetectionPipeline(camera, y_mpp, x_mpp)

clip = "project_video"
clip_output_path = "./{}_output.mp4".format(clip)
input_clip = VideoFileClip('./{}.mp4'.format(clip))
output_clip = input_clip.fl_image(lambda img: pipeline.process(img))
output_clip.write_videofile(clip_output_path, audio=False)
