import glob
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2)
objpoints, imgpoints = [], []


def camera_calibration(image_path):
    image = plt.imread(image_path)
    image_path = image_path.split('/')[-1]
    print(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1::-1], None, None)
        des = cv2.undistort(image, mtx, dist)

        offset = 100
        img_size = (gray.shape[1], gray.shape[0])
        src = np.float32([corners[0], corners[8], corners[-1], corners[-9]])
        dst = np.float32(
            [[offset, offset], [img_size[0] - offset, offset], [img_size[0] - offset, img_size[1] - offset],
             [offset, img_size[1] - offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        M_ = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(des, M, img_size)
        # fig = plt.figure()
        plt.imshow(warped)
        plt.savefig("./output_images/"+image_path)
        print("ojbk")


image_path = glob.glob("./camera_cal/calibration*.jpg")
camera_calibration(image_path[3])
