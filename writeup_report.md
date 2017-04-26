# Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[video1]: ./project_video.mp4 "Video"
[undistorted_image1]: ./output_images/camera_calibration.png "Camera calibration"
[undistorted_image2]: ./output_images/undistorted.jpg "Undistorted"
[original_image2]: ./test_images/straight_lines2.jpg "Original"
[binary_image]: ./output_images/binary.jpg "Binary"
[src_rect]: ./output_images/before_warp_with_src_rect.jpg "Source Rect"
[dst_rect]: ./output_images/after_warp_with_dst_rect.jpg "Destination Rect"
[warped_masking]: ./output_images/warped_masking.jpg "Warped Masking"
[unwarped_masking]: ./output_images/unwarped_masking.jpg "Unwarped Masking"
[annotated_result]: ./output_images/annotated_result.jpg "Annotated Result"

# [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## Camera Calibration

### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 16 - 46 of the file called `camera.py`.

To do camera calibration, I am first preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. I assumed the chessboard is fixed on the (x, y) plane at z=0, such that object points are always the same for each calibration image. I stored the prepared "object points" in the variable `objpoints`.

Next I used `cv2.findChessboardCorners()` function to locate the corners of the chessboard. If corners are not found (the first return value from `cv2.findChessboardCorners` is None), I simply skipped the image and continue with the next one; if corners are found, I stored the results into the array `imgp`, and also make copy of `objpoints` and appended to the array `objp` to keep the mapping amount them.

After visiting all the calibration images, I then used the collected `imgp` and `objp` the compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The camera matrix and distortion coefficient are then stored for future use.

By invoking `undistort()` method in the `Camera` class (line 48 - 53 in `camera.py`), which in turns calling the `cv2.undistort()` function, I can repeatedly perform camera correction on input image. Here's an example when I applied camera correction on a test image:

![Camera correction][undistorted_image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Here're examples before applying distortion correction and that of applied:
![Before correction image][original_image2]
![Undistorted image][undistorted_image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. Thresholding steps
can be found at lines 292 - 324 in `processing.py`. To summarize briefly, gaussian blur filter will be applied to the image first to reduce nosies. After that, thresholds the v value in HSV color space and s value in HSL color space to a specific range. Meanwhile, Sobel operator in x direction is applied on the v channel. Finally, combine all of these thresholds to get the final binary image.

Here's an example of my output for this step.

![Binary image][binary_image]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes:
 
1. A class method called `setup_perspective_transform()` in class `Camera`, which appears in lines 55 - 59 in the file `camera.py`. This method is used to specify the source rect (as the formal argument `src_rect`) before perspective transform and destination rect (as the formal argument `dst_rect`) after perspective transform.

2. A class method called `warp_perspective()` in class `Camera`, which appears in lines 61 - 65 in the file `camera.py`. This method is used to perform perspective transform on any given image.

3. A class method called `warp_inverse_perspective()` in class `Camera`, which appears in line 67 - 70 in the file `camera.py`. This method is used to perform inverse perspective transform on any given image.

Once perspective transform is setup via `setup_perspective_transform()`, the transform matrix and inverse transform matrix will be remembered in `Camera` instance for later use.

The perspective transform matrix is assumed to be unchanged throughout the project video. `src_rect` and `dst_rect` are hardcoded in the following manner:

```
src_rect = np.array(((595, 447),
                     (237, 697),
                     (1085, 697),
                     (686, 447)), dtype=np.float32)
dst_rect = np.array(((300, 0),
                     (300, 720),
                     (980, 720),
                     (980, 0)), dtype=np.float32)
```

This resulted in the following mapping:


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 447      | 300, 0        | 
| 237, 697      | 300, 720      |
| 1085, 697     | 980, 720      |
| 686, 447      | 980, 0        |

I verified that my perspective transform was working as expected by drawing the `src_rect` and `dst_rect` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

This is the example binary image with source rect drawn on it:

![With Source Rect][src_rect]

This is the example warped binary image with destination rect drawn:

![With Destination Rect][dst_rect]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After warping the extracted binary image into a bird-eye perspective view, I used a sliding window search algorithm to identify pixels for a lane. The whole processing procedure can be found in class method `process()` in class `LaneDetectionPipeline` in lines 36 - 121 for file `pipeline.py`. It is consisted of several parts:

1. The process of extracting bird-eye view binary image, in lines 39 - 45.

2. The process for finding lane pixels, in lines 49 - 54. The process can again divided into two parts:

    1. When no fitting information for last few frames is available, the pipeline will perform a full sliding window search. My implementation of sliding window search is much similar to the implementation in the course video. I used the convolution approach since it's more concise. Codes for sliding window search can be found in lines 8 - 148 in `processing.py`.
    
    2. When fitting information for last few frames are available, the pipeline will search the pixels for the lanes of current frame by using the polynomial fit in the last frame. It will first by sampling lane center points from the polynomial, and use the center points to do pixeles search. Codes for this can be found in lines 151 - 196 in `processing.py`.
    
   In either case the points found will be returned. If left lane/right lane cannot be found by this step, the points found and polynomial fit in previous frame will be used, skipping step 3. If those are not available, then the pipeline will treat the current frame as a hard or a bad one and give up finding. The process for the current frame ends.

3. Fitting a 2nd order polynomial to describe the points found in step 2. Related codes can be found in lines 78 - 79 in `pipeline.py`. Codes doing polynomial fit can be found in lines 199 - 202 in `processing.py`.

4. Averaging polynomial for current frame with those in the previous frames. The way I used to averaging polynomial is, by sampling points from the given polynomials with different weights, then fitting a new polynomial for these points. Detailed codes can be found in file `fitting.py`.

5. Creating a masking imaged which is used to combined with the undistorted image and unwarp, to produce the annotated output. Codes can be found in lines 92 - 98.

Here's an example for the warped masking image:

![Warped masking image][warped_masking]

and the unwarped masking image:
![Unwarped masking image][unwarped_masking]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 226 - 233 in the file `processing.py`. The main idea is converting each points in pixel to points in meter then fit again. Depending on the `src_rect` I picked for perspective transform, I set the meter in pixel in x direction to be (3.7 / (980 - 300)), meter in pixel in y direction to be (30 / 720).

After fitting polynomials in meters, curvatures are calculated using the formulas denoted in course videos.

Average of the curvature of the left lane and right lane is taken and used as the curvature of the road.

The center of the detected lanes is calculated too and compared with the center of the image. The difference is then converted into meters. This can be viewed as the off-center distance for the vehicle.

Codes for this step can be found in lines 101 - 112 in `pipeline.py`.
 
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 205 - 233 in `processing.py` in the function `get_birdview_lane_mask_image()` and in lines 92 - 98 in `pipeline.py`.  Here is an example of my result on a test image:

![Annotated result][annotated_result]

---

### Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think the tricky part for this project is how to find out a good way to threshold the input image to a binary one. Although my combination is good enough to tackle with the project video, it do not perform well in the other challenge videos. Different lighting conditions may cause my s threshold and v threshold throwing away too much details, which result in few edges are detected. Meanwhile, the Sobel operator can yielding edges for not only lanes but also significant changing of colors, such as a dark shadow, which result in too many noise features. All of these will make my pipeline failed.
  
To make my pipeline more robust, I think in addition to trying more combination of thresholding, I can try to implement outliers detection in my pipeline so as to make the pipeline more resist to noise. Although I don't have a clear picture about how to do this, may be I can use the fact that I'm trying to fit a curve. Any pixels contributing to a drastic change in curvature might be denoted as outliers.
   
Another direction I think I can work on is to improve the sliding window search procedure. In my current pipeline, the center of searching window for the next slice is the same as the center detected in the current slice, but unless the lane to be found is actually a straight line, the center will surely change a bit from slice to slice. Leveraging this might make my pipeline more robust to noise too.
