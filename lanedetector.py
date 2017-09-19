import numpy as np
import cv2
import glob

class LaneDetector:
    def __init__(self):
        self.points_per_row = 9
        self.points_per_col = 6
        self.cameraMatrix = None
        self.distortionCoeffs = None
        self.perspectiveTransformMatrix = None
        self.srcPoints = None
        self.dstPoints = None

    def _calibrate(self):
        objp = np.zeros((self.points_per_row * self.points_per_col, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.points_per_row, 0:self.points_per_col].T.reshape(-1,2)
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('camera_cal/*.jpg')
        for idx, filename in enumerate(images):
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.points_per_row, self.points_per_col), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, self.cameraMatrix, self.distortionCoeffs, rvecs, tvecs \
                                    = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # print('self.cameraMatrix: {}'.format(self.cameraMatrix))
        # print('self.distortionCoeffs: {}'.format(self.distortionCoeffs))

    def undistort(self, img):
        if self.cameraMatrix is None:
            self._calibrate()
        return cv2.undistort(img, self.cameraMatrix, self.distortionCoeffs, None, self.cameraMatrix)

    def _calculate_perspective_transform_matrix(self):
        filename = 'test_images/straight_lines1.jpg'
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_size = (gray.shape[1], gray.shape[0])
        self.srcPoints = np.float32(
                        [[img_size[0] * 0.151, img_size[1]],
                        [img_size[0] * 0.451, img_size[1] * 0.64],
                        [img_size[0] * 0.553, img_size[1] * 0.64],
                        [img_size[0] * 0.888, img_size[1]]])
        self.dstPoints = np.float32(
                        [[(img_size[0] / 4), img_size[1]],
                        [(img_size[0] / 4), 0],
                        [(img_size[0] * 3 / 4), 0],
                        [(img_size[0] * 3 / 4), img_size[1]]])
        # print('srcPoints: {}'.format(np.int32(self.srcPoints)))
        # print('dstPoints: {}'.format(np.int32(self.dstPoints)))
        self.perspectiveTransformMatrix = cv2.getPerspectiveTransform(self.srcPoints, self.dstPoints)

    def warp_perspective(self, img):
        if self.perspectiveTransformMatrix is None:
            self._calculate_perspective_transform_matrix()
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.perspectiveTransformMatrix, img_size, flags=cv2.INTER_LINEAR)

    def generate_binary_image(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Stack each channel
        #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

    def find_lanelines(self, img):
        pass

    def measure_curvature(self, img):
        pass
