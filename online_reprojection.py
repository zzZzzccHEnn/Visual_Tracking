from pymycobot import MyCobotSocket
from pymycobot import PI_PORT, PI_BAUD
import cv2
import numpy as np
from time import gmtime, strftime
from forward_kinematics import F_K
from inverse_kinematics import I_K

class online_reproject():
    def __init__(self):
        #####################CONNECTION##########################
        self.robot = MyCobotSocket("172.25.128.1",9000)
        self.robot.connect("/dev/ttyAMA0","1000000")
        self.robot.power_on()
        #####################CONNECTION##########################
        
        # determind the camera
        self.cap = cv2.VideoCapture(2)

        # load the T_mar2base
        self.T_mar2base = np.load("scripts/Analysis"
                                  + "/T_mar2base_procrustes.npy")

        # load the T_cam2ee
        T_cam2ee = np.load("scripts/Hand_eye_calibration"
                           + "/cam2gripper.npy")
        self.T_ee2cam = np.linalg.inv(T_cam2ee)

        # create the object points of the chessboard (9X6)
        self.width = 9
        self.height = 6
        self.objp = np.zeros((self.height*self.width,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.width,
                                   0:self.height].T.reshape(-1,2) * 0.016

        # load the camera parameter
        self._mtx = np.load("scripts/Hand_eye_calibration"
                            + "/camera_matrix.npy")
        self._dist = np.load("scripts/Hand_eye_calibration"
                             + "/distortion_coefficients.npy")


    def run(self):

        # initial pose
        self.robot.sync_send_angles([0,0,0,0,0,0], 30)
        self.robot.release_all_servos()

        ####################################################################

        while self.cap.isOpened():
            success, img = self.cap.read()
            if success:
                # monitor the image from camera, ensure the chessboard 
                # in the image so that the reprojection will work
                cv2.imshow("camera", img)
                cv2.waitKey(1)

                # store the image 
                img_opencv = img.copy()
                img_reproject = img.copy()

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                #### detect the corners from OpenCV ####
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(
                                                    gray, 
                                                    (self.width, self.height), 
                                                     None
                                                     )

                # if corners are detected then
                if ret:
                    # convert the pixel coordinates to integers
                    int_corners_opencv = [np.squeeze(i).astype(np.int64) 
                                          for i in corners]

                    #draw the corners detected from opencv
                    for i in range(len(int_corners_opencv)):
                        # draw the points overlaying the original image
                        img_opencv = cv2.circle(img_opencv, 
                                                int_corners_opencv[i], 
                                                radius=2, color=(0, 255, 0), 
                                                thickness=-1)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_1 = "corners detected from OpenCV"
                    font_scale = 0.8
                    img_opencv = cv2.putText(img_opencv, text_1, 
                                             (130,30), 
                                             font, font_scale, 
                                             (0, 255, 0), 
                                             1, cv2.LINE_AA)
                        
                    ###########################################################
                    # project the corners via kinematics
                    # create an object of inverse kinematics class
                    ik = I_K()
                    joint_angles = self.robot.get_angles()
                    # forward kinematics T_ee2base
                    T_ee2base = ik.forward(joint_angles)
                    T_base2ee = np.linalg.inv(T_ee2base)

                    # calculate T_mar2cam
                    T_mar2cam = self.T_ee2cam @ T_base2ee @ self.T_mar2base

                    # project the chessboard corners to the camera frame
                    re_projected_points, _ = cv2.projectPoints(
                                                np.array(self.objp), 
                                                T_mar2cam[0:3,0:3], 
                                                T_mar2cam[0:3,3], 
                                                self._mtx, 
                                                self._dist)
                    
                    # convert the pixel coordinates to integers
                    int_reproject_points = [np.squeeze(i).astype(np.int64) 
                                            for i in re_projected_points]

                    #### project the points ####
                    for i in range(len(int_reproject_points)):
                        # draw the points overlaying the original image
                        img_reproject = cv2.circle(img_reproject, 
                                                   int_reproject_points[i], 
                                                    radius=2, 
                                                    color=(0, 0, 255), 
                                                    thickness=-1)
                    
                    text_2 = "corners reprojected"
                    img_reproject = cv2.putText(img_reproject, text_2, 
                                                (180,30), 
                                                font, 
                                                font_scale, 
                                                (0, 0, 255), 
                                                1, cv2.LINE_AA)
                        
                    # calculate the error (pixel)
                    error = np.sqrt(
                            np.sum(
                            np.square(
                            np.array(int_corners_opencv)
                              - np.array(int_reproject_points)))) / (9*6)
                    
                    error = np.round(error, decimals=5)
                    text_error = "Error = " + str(error) + " pixel(s)"
                    # display the error 
                    img_reproject = cv2.putText(img_reproject, 
                                                text_error, (180,460), 
                                                font, font_scale, 
                                                (0, 0, 255), 1, 
                                                cv2.LINE_AA)

                    # merge the images
                    merged_img = cv2.hconcat((img_opencv, img_reproject))
                    # show the image 
                    # cv2.imshow("opencv", img_opencv)
                    # cv2.imshow("reprojection", img_reproject)
                    cv2.imshow("online_reprojection", merged_img)
                    # change the waitkey to control the fps
                    k = cv2.waitKey(1)

                    # press "esc" to exit
                    if k == 27:
                        break
                    # press "s" to save the figure
                    elif k == ord("s"):
                        now = strftime("%d-%m-%Y_%H:%M:%S", gmtime())
                        cv2.imwrite("scripts/Analysis/online_reprojection/" 
                                    + now + ".png", merged_img)


if __name__ == "__main__":
    reproject = online_reproject()
    reproject.run()