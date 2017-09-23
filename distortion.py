import cv2
import os
import pickle
import numpy as np

class CameraCalibrator:
    def __init__(self, camera_cal_dir = 'camera_cal/', points_per_row = 9, points_per_col = 6):
        self.camera_cal_dir = camera_cal_dir
        self.points_per_row = points_per_row
        self.points_per_col = points_per_col

        self.cameraMatrix = None
        self.distortionCoeffs = None

        if os.path.exists('distortion.pickle'):
            with open('distortion.pickle', 'rb') as f:
                self.cameraMatrix, self.distortionCoeffs = pickle.load(f)

        if self.cameraMatrix is None or self.distortionCoeffs is None:
            print('No valid pickle file found. Calibrating ...')
            self._calibrate()
            print('Done')
        else:
            print('Load caerma matrix and distortion coeffs from pickle file.')
        # print('self.cameraMatrix: {}'.format(self.cameraMatrix))
        # print('self.distortionCoeffs: {}'.format(self.distortionCoeffs))

    def _calibrate(self):
        objp = np.zeros((self.points_per_row * self.points_per_col, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.points_per_row, 0:self.points_per_col].T.reshape(-1,2)
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(self.camera_cal_dir + '*.jpg')
        for idx, filename in enumerate(images):
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.points_per_row, self.points_per_col), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, self.cameraMatrix, self.distortionCoeffs, rvecs, tvecs \
                                    = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # Write to pickle file
        with open('distortion.pickle', 'wb') as f:
            pickle.dump([self.cameraMatrix, self.distortionCoeffs], f)

    def undistort(self, img):
        if self.cameraMatrix is None:
            self._calibrate()
        return cv2.undistort(img, self.cameraMatrix, self.distortionCoeffs, None, self.cameraMatrix)
