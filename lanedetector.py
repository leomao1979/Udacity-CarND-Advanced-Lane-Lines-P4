import numpy as np
import cv2
import glob
import scipy
from distortion import CameraCalibrator
from transformer import PerspectiveTransformer

class LaneDetector:
    def __init__(self):
        self.leftx = None
        self.lefty = None
        self.rightx = None
        self.righty = None
        self.left_fit = None
        self.right_fit = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.left_fits = []
        self.right_fits = []

        self.margin = 100
        self.minpix = 50            # minimum pixels to recenter window
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

        self.cameraCalibrator = CameraCalibrator()
        self.perspectiveTransformer = PerspectiveTransformer()

    def test_binary_image(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        # sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        sxbinary[scaled_sobel > 0] = 255

        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        yellow_hsv_low = np.array([0, 80, 180], np.uint8)
        yellow_hsv_high = np.array([40, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsv_img, yellow_hsv_low, yellow_hsv_high)

        white_rgb_low = np.array([200, 200, 200], np.uint8)
        white_rgb_high = np.array([255, 255, 255], np.uint8)
        white_mask = cv2.inRange(img, white_rgb_low, white_rgb_high)

        yellow_white_mask = cv2.bitwise_or(yellow_mask, white_mask)
        white = cv2.bitwise_and(img, img, mask=yellow_white_mask)
        white_binary = cv2.cvtColor(white, cv2.COLOR_RGB2GRAY)
        white_binary[white_binary > 0] = 1
        return (sxbinary, s_binary)

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
        yellow_hsv_low = np.array([0, 80, 180], np.uint8)
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

    def _slide_windows_to_extract_line_points(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds]
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]

    def _draw_detected_lanes(self, binary_warped):
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img[self.lefty, self.leftx] = [255, 0, 0]
        out_img[self.righty, self.rightx] = [0, 0, 255]
        out_img[np.int_(ploty), np.int_(left_fitx)] = [255, 255, 0]
        out_img[np.int_(ploty), np.int_(right_fitx)] = [255, 255, 0]

        return out_img

    def _calculate_line_points(self, binary_warped):
        # Skip sliding windows to search in a margin around the previous line position
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_fit_x = self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2]
        right_fit_x = self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2]
        left_lane_inds = ((nonzerox > (left_fit_x - self.margin)) & (nonzerox < (left_fit_x + self.margin)))
        right_lane_inds = ((nonzerox > (right_fit_x - self.margin)) & (nonzerox < (right_fit_x + self.margin)))

        # Extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds]
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]

    def _fill_color_between_lanes(self, undistorted, warped):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.perspectiveTransformer.Minv, (undistorted.shape[1], undistorted.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.5, 0)

        lane_img = np.dstack((warp_zero, warp_zero, warp_zero))
        lane_img[self.lefty, self.leftx] = [255, 0, 0]
        lane_img[self.righty, self.rightx] = [0, 0, 255]
        lane_img[np.int_(self.ploty), np.int_(self.left_fitx)] = [255, 255, 0]
        lane_img[np.int_(self.ploty), np.int_(self.right_fitx)] = [255, 255, 0]
        sub_window_size = (np.int_(lane_img.shape[0] / 3), np.int_(lane_img.shape[1] / 3))
        lane_img = scipy.misc.imresize(lane_img, sub_window_size)
        result[:sub_window_size[0], -sub_window_size[1]:] = lane_img

        color_warped = np.dstack((warped, warped, warped)) * 255
        color_warped = scipy.misc.imresize(color_warped, sub_window_size)
        result[:sub_window_size[0], -2*sub_window_size[1]:-sub_window_size[1]] = color_warped

        return result

    def _update_fit(self, img_size):
        self.ploty = np.linspace(0, img_size[1]-1, img_size[1])
        if len(self.leftx) > 0:
            self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
            self.left_fits.append(self.left_fit)
            self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            self.left_fitx[self.left_fitx < 0] = 0
            self.left_fitx[self.left_fitx >= img_size[0]] = img_size[0]-1
        if len(self.rightx) > 0:
            self.right_fit = np.polyfit(self.righty, self.rightx, 2)
            self.right_fits.append(self.right_fit)
            self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
            self.right_fitx[self.right_fitx < 0] = 0
            self.right_fitx[self.right_fitx >= img_size[0]] = img_size[0]-1

    def detect_lanes(self, img):
        undistorted = self.cameraCalibrator.undistort(img)
        binary_image = self.generate_binary_image(undistorted)
        binary_warped = self.perspectiveTransformer.warp_perspective(binary_image)
        if self.left_fit is None:
            self._slide_windows_to_extract_line_points(binary_warped)
        else:
            self._calculate_line_points(binary_warped)
        # Update left / right fit
        self._update_fit((binary_warped.shape[1], binary_warped.shape[0]))
        result = self._fill_color_between_lanes(undistorted, binary_warped)
        sub_window_size = (np.int_(result.shape[0] / 3), np.int_(result.shape[1] / 3))
        color_image = np.dstack((binary_image, binary_image, binary_image)) * 255
        color_image = scipy.misc.imresize(color_image, sub_window_size)
        result[:sub_window_size[0], :sub_window_size[1]] = color_image

        left_curverad, right_curverad = self._measure_curvature()
        curve_text = 'Radius of Curvature: ({}m, {}m)'.format(round(left_curverad, 1), round(right_curverad, 1))
        cv2.putText(result, curve_text, (50, sub_window_size[0] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return result

    def _measure_curvature(self):
        y_eval = 720
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty*self.ym_per_pix, self.left_fitx*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*self.ym_per_pix, self.right_fitx*self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        return (left_curverad, right_curverad)
        # Example values: 632.1m, 626.2m
