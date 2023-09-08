import cv2
import numpy as np
import time
import glob
import os
import re
import matplotlib.pyplot as plt

class Camera_Calibration:
    def __init__(self):
        # create the dictionary to store the image for calibration
        self._path = os.getcwd() + "/cam_calibration_images"
        if not os.path.exists(self._path):
            os.mkdir(self._path)
        
        # determind the camera
     #    self.cap = cv2.VideoCapture(2)

    def camera_calibration_validate(self):
        # print("###########Waiting for Calibration###########")

        width = 9
        height = 6

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((height*width,3), np.float32)
        objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2) * 0.016

        # if mode == "cam":
        #     images = sorted(glob.glob('scripts/cam_calibration_images/*.png'))
        
        # else:
        #     images = sorted(glob.glob('scripts/images/*.png'))
        
        images = sorted(glob.glob('scripts/cam_calibration_images/*.png'))

        print(len(images), "in total")
        
        error = []
        numb_images = []
        
        for n in range(0,25):
                # intialise the object points and image points
                objpoints = [] # 3d point in real world space
                imgpoints = [] # 2d points in image plane.
                # if n == 1:
                #     calibrate_images = images[0:1]
                # else:
                #     calibrate_images = images[0:n-1]
                calibrate_images = images[0:n+1]
                numb_images.append(n+1)

                for fname in calibrate_images:
                    img = cv2.imread(fname)
                    _name = str(fname)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Find the chess board corners
                    ret, corners = cv2.findChessboardCorners(gray, (width,height), None)

                    # If found, add object points, image points (after refining them)
                    if ret == True:
                         objpoints.append(objp)
                         corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                         imgpoints.append(corners2)
                
                print("len of objpoints", len(objpoints))

                self.ret, self.mtx, self.dist, self.rvecs, self.tvecs, self.stdDeviationsIntrinsics, self.stdDeviationsExtrinsics, self.perViewErrors = cv2.calibrateCameraExtended(objpoints, imgpoints, 
                                                                                                                                                                                    gray.shape[::-1], 
                                                                                                                                                                                    None, None)
            
                if len(calibrate_images) != len(self.rvecs):
                    print(len(calibrate_images) - len(self.rvecs), "image(s) calibration failed")
                
                else:
                    error.append(np.average(self.perViewErrors))
                    print("average error is", np.average(self.perViewErrors))
                    print("###########Calibration Done###########")
                    print()

        plt.plot(np.arange(1,26), error)
        # plt.xticks(range(1,26))
        plt.savefig("average_error.png")
        plt.show()

if __name__ == "__main__":
    calibration = Camera_Calibration()
    calibration.camera_calibration_validate()
    # print(np.arange(1,20))
    # plt.xticks(range(1,3))
    # plt.show()