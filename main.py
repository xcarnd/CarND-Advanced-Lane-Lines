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
dst_rect = np.array(((300, 0),
                     (300, 720),
                     (980, 720),
                     (980, 0)), dtype=np.float32)

camera.setup_perspective_transform(src_rect, dst_rect)

y_mpp = 30 / 720
x_mpp = 3.7 / (980-300)
import cv2
import matplotlib.pyplot as plt

# img0000 = cv2.imread("./seq1/frame0000.jpg")
# img0000 = cv2.cvtColor(img0000, cv2.COLOR_BGR2RGB)
# out0000 = pipeline.process(img0000)
#
# img0001 = cv2.imread("./seq1/frame0001.jpg")
# img0001 = cv2.cvtColor(img0001, cv2.COLOR_BGR2RGB)
# out0001 = pipeline.process(img0001)
#
# f, ((ax11, ax12), (ax21, ax22))= plt.subplots(2, 2)
# ax11.imshow(img0000)
# ax12.imshow(out0000)
# ax21.imshow(img0001)
# ax22.imshow(out0001)
# plt.show()

pipeline = LaneDetectionPipeline(camera, y_mpp, x_mpp)

clip = "project_video"
clip_output_path = "./{}_output.mp4".format(clip)
input_clip = VideoFileClip('./{}.mp4'.format(clip))
output_clip = input_clip.fl_image(lambda img: pipeline.process(img))
output_clip.write_videofile(clip_output_path, audio=False)
# input_clip.write_images_sequence("seq1/frame%04d.jpg")
# output_clip.write_images_sequence("debug/frame%04d.jpg")


# import cv2
# import matplotlib.pyplot as plt
#
# for file_name in os.listdir("./test_images"):
#     input_path = "./test_images/" + file_name
#     output_path = "./output_images/" + file_name
#     print(input_path, output_path)
#     img = cv2.imread(input_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     pipeline = LaneDetectionPipeline(camera, y_mpp, x_mpp)
#     output = pipeline.process(img)
#     cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
#     plt.imshow(output)
#     plt.show()
