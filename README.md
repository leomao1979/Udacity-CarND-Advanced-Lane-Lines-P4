# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  

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

[undistorted_cal1]: ./output_images/undistort_calibration1.jpg "Undistorted"
[undistorted_test1]: ./output_images/undistort_test1.jpg "Undistorted Test1"
[binary_test1]: ./output_images/binary_test1.jpg "Binary Test1"
[binary_challenge1]: ./output_images/binary_challenge1.jpg "Binary Challenge1"
[warped]: ./output_images/warped_straight_lines1.jpg "Warp Example"
[undistorted_random]: ./output_images/undistorted_random.jpg "Undistorted Random"
[binary_random]: ./output_images/binary_random.jpg "Binary Random"
[warped_random]: ./output_images/warped_line_random.jpg "Warped Line Random"
[result_random]: ./output_images/result_random.jpg "Result  Random"
[video1]: ./output_videos/detect_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration and image undistortion are implemented in class CameraCalibrator (distortion.py).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Undistorted][undistorted_cal1]

The calibrated camera matrix and distortion coefficients are saved to pickle file for future use.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is an example of distortion correction to one of the test images:

![Undistorted][undistorted_test1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I tried different combinations of color and gradient thresholds to generate a binary image (lines 82 through 125 in `main.py`), while the result of thresholded gradients is not impressive. After comparison I decided to use color threshold only (lines 44 through 58 in `lanedetector.py`). Here are two examples to compare different choices:

Example 1:

![Binary 1][binary_test1]

Example 2:

![Binary 2][binary_challenge1]

My final choice is to detect yellow line with HSV and white line with RGB. Here's an example of my output for this step.

**Undistorted Image captured during video processing**

![Undistorted Random][undistorted_random]

**Binary Image generated**

![Binary Random][binary_random]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The class PerspectiveTransformer (transformer.py) is created for perspective transform. It has two methods, `_calculate_matrix()` and `warp_perspective()`. The `_calculate_matrix()` method calculates the transformation matrix `M` and its inverse matrix `Minv`. The `warp_perspective()` takes an image (`img`) as input and uses the matrix `M` to complete the transformation.  I chose hardcode source and destination points as following:

```python
self.srcPoints = np.float32(
                [[img_size[0] * 0.151, img_size[1]],
                [img_size[0] * 0.451, img_size[1] * 0.64],
                [img_size[0] * 0.553, img_size[1] * 0.64],
                [img_size[0] * 0.888, img_size[1]]])

self.dstPoints = np.float32(
                [[(img_size[0] / 4) - 30, img_size[1]],
                [(img_size[0] / 4 - 30), 0],
                [(img_size[0] * 3 / 4 + 30), 0],
                [(img_size[0] * 3 / 4 + 30), img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 193, 720      | 290, 720      |
| 577, 460      | 290, 0        |
| 707, 460      | 990, 0        |
| 1136, 720     | 990, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped][warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

When there is no fitted line available, I use sliding windows to find line pixels (lines 89 through 128 in `line.py`), otherwise I just skip it and search in the margin around the previous line position (lines 81 through 86 in `line.py`).
Then I would check whether there are sufficient line pixels detected. If yes, fit the lines with a 2nd order polynomial and run sanity check on fitted lines.
The newly detected lines would be applied only if they pass the sanity check (lines 167 through 188 in `lanedetector.py` and lines 148 to 164 in `line.py`).

Example:

![Warped Random][warped_random]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I measured radius of curvature in the method `_measure_curvature()` of class Line (lines 140 to 146 in `line.py`) and calculated vehicle position on line 61 in the same class.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the method `_draw_lane()` of class LaneDetector (lines 113 through 131 in `lanedetector.py`). I also added three sub windows on top of the image to demonstrate the process of lane detection (lines 133 through 165 in `lanedetector.py`). Here is an example of my result on a test image:

![Result Random][result_random]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/detect_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline keeps track of last several detections and uses the average value of fitted lane lines to detect new ones (lines 14 through 21 in `line.py`). It will start searching from scratch if last n detections fail.
The pipeline works pretty good with `project_video.mp4` and is acceptable on `challenge_video.mp4` but performs badly with `harder_challenge_video`. The followings need to be  improved to make it more robust.
1) Design a better algorithm to generate binary image. It shall highlight lane lines and reduce noises to improve the detection accuracy.
2) Design an effective sanity check algorithm. The radius of curvature I got is not reliable, so I didn't use it for sanity check. It is critical that a bad detection could be identified and discarded.
