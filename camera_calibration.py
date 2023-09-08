import cv2
import numpy as np
import time
import glob
import os
import re

class Camera_Calibration:
    def __init__(self):
        # create the dictionary to store the image for calibration
        self._path = os.getcwd() + "/cam_calibration_images"
        if not os.path.exists(self._path):
            os.mkdir(self._path)
        
        # determind the camera
        self.cap = cv2.VideoCapture(2)

    def get_chess_board(self):
        # count the images number
        num = 0
        while self.cap.isOpened():
            success, img = self.cap.read()
            if success:
                k = cv2.waitKey(5)
                cv2.imshow("img", img)
                if k==27: # "esc" to end the process
                    break
                elif k == ord('s'): # wait for 's' key to save and exit
                    cv2.imwrite(self._path + '/img' + str(num+1) + '.png', img)
                    print(num+1,"image saved!")
                    num += 1
                    
        cv2.destroyAllWindows()
        print("###########Images Collection Finished###########")

    def camera_calibration(self, mode):
        # print("###########Waiting for Calibration###########")

        width = 9
        height = 6

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((height*width,3), np.float32)
        objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2) * 0.016

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # if mode == "cam":
        #     images = sorted(glob.glob('scripts/cam_calibration_images/*.png'))
        
        # else:
        #     images = sorted(glob.glob('scripts/images/*.png'))
        
        images = sorted(glob.glob('scripts/images/*.png'))
        
        # for item in images:
        #     print(item)

        print(len(images), "in total")
        count = 0
        

        for fname in images:
            img = cv2.imread(fname)
            _name = str(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (width,height), None)

            if ret == False:
                print("Fail:" + fname)

            # If found, add object points, image points (after refining them)
            else:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                count += 1
                # print(count, "images calibrated")
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (width,height), corners2, ret)
                if mode == "hand_eye":
                    calibrated_image = re.sub("scripts/images", "scripts/images/images_calibrated", _name)
                    # print(calibrated_image)
                    cv2.imwrite(calibrated_image, img)
                
                # cv2.imshow(_name, img)
                # cv2.waitKey(100)
                # cv2.destroyWindow(_name)
                # print(_name)
        cv2.destroyAllWindows()
        
        if mode == "cam":
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                                                        gray.shape[::-1], 
                                                                                        None, None)
            

            if len(images) != len(self.rvecs):
                print(len(images) - len(self.rvecs), "image(s) calibration failed")
                
            else:
                np.save("scripts/Hand_eye_calibration/camera_matrix", self.mtx)
                np.save("scripts/Hand_eye_calibration/distortion_coefficients", self.dist)
                print("###########Calibration Done###########")
                print()
        
        elif mode == "hand_eye":
            camera_matrix = np.load("scripts/Hand_eye_calibration/camera_matrix.npy")
            distortion = np.load("scripts/Hand_eye_calibration/distortion_coefficients.npy")

            self.rvecs = []
            self.tvecs = []

            for i in range(len(objpoints)):
                _, _rvecs, _tvecs = cv2.solvePnP(np.array(objpoints[i]), np.array(imgpoints[i]), 
                                                 camera_matrix, distortion)
                self.rvecs.append(_rvecs)
                self.tvecs.append(_tvecs)

            if len(images) != len(self.rvecs):
                print(len(images) - len(self.rvecs), "image(s) calibration failed")
        # self.rvecs and self.tvecs represent the rotation matrices and translation matrices of each image
            else:
                print("###########SlovePnP Done###########")
                print()


def main():
    calibration = Camera_Calibration()
    # get the chess board images
    # calibration.get_chess_board()
    # time.sleep(5)
    # calibration
    calibration.camera_calibration(mode="hand_eye")
    # print("mtx from calibration is",calibration.mtx)
    # print("dist from calibration is",calibration.dist)
    # store the calibration result
    

if __name__ == "__main__":
    main()
