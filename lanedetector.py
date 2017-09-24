import numpy as np
import cv2
import glob
import scipy
from distortion import CameraCalibrator
from transformer import PerspectiveTransformer
from line import Line

class LaneDetector:
    def __init__(self):
        self.cameraCalibrator = CameraCalibrator()
        self.perspectiveTransformer = PerspectiveTransformer()
        self.left_line = Line()
        self.right_line = Line()

    def generate_binary_image(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        # Convert to HLS color space and separate the V channel
        # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        # l_channel = hls[:,:,1]
        # s_channel = hls[:,:,2]
        # Sobel x
        # sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        # abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        # scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        # Threshold x gradient
        # sxbinary = np.zeros_like(scaled_sobel)
        # sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        # s_binary = np.zeros_like(s_channel)
        # s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # combined_binary = np.zeros_like(s_binary)
        # combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        yellow_hsv_low = np.array([0, 60, 160], np.uint8)
        yellow_hsv_high = np.array([40, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsv_img, yellow_hsv_low, yellow_hsv_high)

        white_rgb_low = np.array([200, 200, 200], np.uint8)
        white_rgb_high = np.array([255, 255, 255], np.uint8)
        white_mask = cv2.inRange(img, white_rgb_low, white_rgb_high)
        yellow_white_mask = cv2.bitwise_or(yellow_mask, white_mask)

        white_yellow = cv2.bitwise_and(img, img, mask=yellow_white_mask)
        white_yellow_binary = cv2.cvtColor(white_yellow, cv2.COLOR_RGB2GRAY)
        white_yellow_binary[white_yellow_binary > 0] = 1

        return white_yellow_binary

    # For debug
    def _draw_sliding_windows(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        minpix = 40
        margin = 80
        # Choose the number of sliding windows
        nwindows = 9
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        return out_img

    def _draw_lane(self, undistorted, warped):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, undistorted.shape[0]-1, undistorted.shape[0])
        pts_left = np.array([np.transpose(np.vstack([self.left_line.bestx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.bestx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.perspectiveTransformer.Minv, (undistorted.shape[1], undistorted.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

        return result

    def _draw_sub_windows(self, result, binary_image, warped):
        top_gap = 5
        gap_between_windows = 8
        sub_window_size = (np.int_(result.shape[0]/3), np.int_((result.shape[1]-gap_between_windows*4)/3))

        # Binary image sub window
        start_x = gap_between_windows
        color_image = np.dstack((binary_image, binary_image, binary_image)) * 255
        color_image = scipy.misc.imresize(color_image, sub_window_size)
        result[top_gap:sub_window_size[0]+top_gap, start_x:sub_window_size[1]+start_x] = color_image
        start_x += sub_window_size[1] + gap_between_windows

        # Warped image sub window
        # color_warped = np.dstack((warped, warped, warped)) * 255
        color_warped = self._draw_sliding_windows(warped)
        color_warped = scipy.misc.imresize(color_warped, sub_window_size)
        result[top_gap:sub_window_size[0]+top_gap, start_x:sub_window_size[1]+start_x] = color_warped
        start_x += sub_window_size[1] + gap_between_windows

        # Warped image with lane plotted sub window
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        lane_img = np.dstack((warp_zero, warp_zero, warp_zero))
        ploty = np.linspace(0, result.shape[0]-1, result.shape[0])
        lane_img[self.left_line.ally, self.left_line.allx] = [255, 0, 0]
        lane_img[self.right_line.ally, self.right_line.allx] = [0, 0, 255]
        lane_img[np.int_(ploty), np.int_(self.left_line.bestx)] = [255, 255, 0]
        lane_img[np.int_(ploty), np.int_(self.right_line.bestx)] = [255, 255, 0]
        lane_img = scipy.misc.imresize(lane_img, sub_window_size)
        result[top_gap:sub_window_size[0]+top_gap, start_x:sub_window_size[1]+start_x] = lane_img

        return result

    def detect_lane(self, img):
        undistorted = self.cameraCalibrator.undistort(img)
        binary_image = self.generate_binary_image(undistorted)
        binary_warped = self.perspectiveTransformer.warp_perspective(binary_image)

        # Detect left and right lines
        self.left_line.detect(binary_warped, isLeft=True)
        self.right_line.detect(binary_warped, isLeft=False)

        # Draw detected lane and sub windows
        result = self._draw_lane(undistorted, binary_warped)
        result = self._draw_sub_windows(result, binary_image, binary_warped)

        # Draw texts
        curve_text = 'Radius of Curvature: ({}m, {}m)'.format(round(self.left_line.radius_of_curvature, 1),
                                                              round(self.right_line.radius_of_curvature, 1))
        cv2.putText(result, curve_text, (50, np.int_(result.shape[0] / 3) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        offset = (self.left_line.line_base_pos - self.right_line.line_base_pos) / 2
        if abs(offset) < 0.1:
            offset_text = 'Vehicle is on the center'
        else:
            offset_side = 'left'
            if offset < 0:
                offset_side = 'right'
            offset_text = 'Vehicle is {}m {} to center'.format(round(abs(offset), 2), offset_side)
        cv2.putText(result, offset_text, (50, np.int_(result.shape[0] / 3) + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return result
