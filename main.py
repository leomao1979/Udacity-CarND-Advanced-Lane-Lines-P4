import cv2
import os
import numpy as np
from lanedetector import LaneDetector
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from distortion import CameraCalibrator
from transformer import PerspectiveTransformer

def undistort_image():
    filename = 'calibration2.jpg'
    img = mpimg.imread('camera_cal/' + filename)
    undistorted = CameraCalibrator().undistort(img)
    save_as = output_dir + 'undistort_' + filename
    show_images(img, 'Original Image', undistorted, 'Undistorted Image', save_as=save_as)

def show_images(img1, title1, img2, title2, cmap=None, save_as=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=15)
    if img2 is not None:
        ax2.imshow(img2, cmap)
        ax2.set_title(title2, fontsize=15)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()

def draw_lines(img, points, top_line = True):
    line_img = np.zeros_like(img, dtype=np.uint8)
    print(line_img.shape)
    if top_line:
        cv2.polylines(line_img, np.int32([points]), isClosed=False, color=[255, 0, 0], thickness=5)
    else:
        cv2.line(line_img, tuple(points[0]), tuple(points[1]), color=[255, 0, 0], thickness=5)
        cv2.line(line_img, tuple(points[2]), tuple(points[3]), color=[255, 0, 0], thickness=5)
    return cv2.addWeighted(img, 0.8, line_img, 1.0, 0.)

def warp_images():
    for filename in filenames:
        img = mpimg.imread(image_dir + filename)
        img = detector.cameraCalibrator.undistort(img)
        warped = detector.perspectiveTransformer.warp_perspective(img)
        img = draw_lines(img, detector.perspectiveTransformer.srcPoints, top_line=True)
        warped = draw_lines(warped, detector.perspectiveTransformer.dstPoints, top_line=False)

        save_as = output_dir + 'warped_' + filename
        show_images(img, 'Undistorted Image with source points drawn', warped,
                         'Warped Image with dest. points drawn', save_as=save_as)

def generate_binary_image():
    for filename in filenames:
        img = mpimg.imread(image_dir + filename)
        img = detector.cameraCalibrator.undistort(img)
        binary_image = detector.generate_binary_image(img)
        save_as = output_dir + 'binary_' + filename
        show_images(img, 'Undistorted Image', binary_image, 'Binary Image', cmap='gray', save_as=save_as)

        warped = detector.perspectiveTransformer.warp_perspective(binary_image)
        binary_image = np.dstack((binary_image, binary_image, binary_image)) * 255
        warped = np.dstack((warped, warped, warped)) * 255
        binary_image = draw_lines(binary_image, detector.perspectiveTransformer.srcPoints, top_line=True)
        warped = draw_lines(warped, detector.perspectiveTransformer.dstPoints, top_line=False)
        save_as = output_dir + 'warped_binary_' + filename
        show_images(binary_image, 'Undistorted Image with source points drawn', warped,
                         'Warped Image with dest. points drawn', save_as=save_as)

def test_binary_image():
    for filename in filenames:
        img = mpimg.imread(image_dir + filename)
        img = detector.cameraCalibrator.undistort(img)
        white_binary, s_binary = detector.test_binary_image(img)
        save_as = output_dir + 'test_sbinary_' + filename
        show_images(img, 'Undistorted Image', s_binary, 'S Binary Image', cmap='gray', save_as=save_as)
        save_as = output_dir + 'test_whitebinary_' + filename
        show_images(img, 'Undistorted Image', white_binary, 'White Binary Image', cmap='gray', save_as=save_as)

def detect_lanes():
    for filename in filenames:
        img = mpimg.imread(image_dir + filename)
        detected = detector.detect_lanes(img)
        plt.imshow(detected)
        plt.show()

def process_image(img):
    result = detector.detect_lanes(img)
    # plt.imshow(result)
    # plt.show()
    return result

def test_video():
    # video_file = 'challenge_video.mp4'
    video_file = 'project_video.mp4'
    clip = VideoFileClip(video_file).subclip(18, 27)
    # clip = VideoFileClip(video_file)
    new_clip = clip.fl_image(process_image)
    video_output = 'output_videos/detect_' + video_file
    new_clip.write_videofile(video_output, audio=False)

detector = LaneDetector()
image_dir = 'test_images/'
# filenames = ['straight_lines1.jpg', 'straight_lines2.jpg', 'test3.jpg']
#filenames = os.listdir(image_dir)
filenames=['challenge1.jpg']
output_dir = 'output_images/'
# test_binary_image()
#undistort_image()
#warp_images()
#generate_binary_image()
detect_lanes()
#test_video()
