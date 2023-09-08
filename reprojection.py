import numpy as np
import time, glob, cv2
from pymycobot import MyCobotSocket
from pymycobot import PI_PORT, PI_BAUD
from single_aruco_detection import marker_detecting
from forward_kinematics import F_K
from inverse_kinematics import I_K
import matplotlib.pyplot as plt


class Reproject():
    def reproject_corners(self, img_number):

        # load the camera parameter
        _mtx = np.load("scripts/Hand_eye_calibration"
                       + "/camera_matrix.npy")
        _dist = np.load("scripts/Hand_eye_calibration"
                        + "/distortion_coefficients.npy")

        # load the T_mar2base
        # T_mar2base = np.load("scripts/Analysis/T_mar2base.npy")
        T_mar2base_procrustes = np.load("scripts/Analysis"
                                        + "/T_mar2base_procrustes.npy")

        # load the T_cam2ee
        T_cam2ee = np.load("scripts/Hand_eye_calibration/cam2gripper.npy")
        T_ee2cam = np.linalg.inv(T_cam2ee)

        # load the T_ee2base
        T_ee2base_file = sorted(glob.glob("scripts/Hand_eye_calibration"
                                          + "/T_gripper2base/*.npy"))
        T_ee2base = [np.load(f) for f in T_ee2base_file]
        T_base2ee = [np.linalg.inv(i) for i in T_ee2base]
    
        # print(T_base2ee[1])

        # transformation from marker to cam
        # through robot kinematics
        # procrustes
        T_mar2cam = T_ee2cam @ T_base2ee[img_number] @ T_mar2base_procrustes


        # print(T_mar2cam)

        # create the object points of the chessboard
        width = 9
        height = 6

        objp = np.zeros((height*width,3), np.float32)
        objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2) * 0.016


        img_file = sorted(glob.glob('scripts/images/*.png'))

        # test for the first image
        img = cv2.imread(img_file[img_number])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_opencv = img.copy()
        img_reproject = img.copy()

        re_projected_points, _ = cv2.projectPoints(np.array(objp), 
                                                   T_mar2cam[0:3,0:3], 
                                                    T_mar2cam[0:3,3], 
                                                    _mtx, _dist)

        # print(re_projected_points)

        int_reproject_points = [np.squeeze(i).astype(np.int64) 
                                for i in re_projected_points]

        # print(int_reproject_points)

    ##########################################################################
    #### project the points ####
        for i in range(len(int_reproject_points)):
            img_reproject = cv2.circle(img_reproject, 
                                       int_reproject_points[i], 
                                       radius=2, color=(255, 0, 0), 
                                       thickness=-1)

    ##########################################################################
    #### detect the corners from OpenCV ####
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (width,height), None)

        if ret == False:
            print("Fail:" + img_file[img_number])
        else:
            int_corners_opencv = [np.squeeze(i).astype(np.int64) 
                                  for i in corners]
            # print(np.array(int_corners_opencv) - np.array(int_reproject_points))

            #draw the corners detected from opencv
            for i in range(len(int_corners_opencv)):
                img_opencv = cv2.circle(img_opencv, 
                                        int_corners_opencv[i], 
                                        radius=2, color=(0, 255, 0), 
                                        thickness=-1)
    
    ##########################################################################
    # calculate the average error of each corners between opencv and reprojection
    # unit of error is "pixel"
    # 9X6 chessboard
        error = np.sqrt(
                np.sum(
                np.square(
                np.array(int_corners_opencv) 
                - np.array(int_reproject_points)))) / (9*6)
        
        error = np.round(error, decimals=5)

    ##########################################################################
        font2 = {'size':16}
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.imshow(img_opencv)
        plt.title("OpenCV",fontdict=font2)
        plt.subplot(1,2,2)
        plt.title("Reprojection", fontdict=font2)
        plt.imshow(img_reproject)
        _text = "error = " + str(error) + " pixel(s)"
        plt.xlabel(_text, fontdict={'size':14})
        plt.tight_layout()
        plt.savefig("scripts/Analysis/reproject_chessboard_corners/_reproject"
                    + str(img_number)+".png", dpi=400)
        # plt.show()


    def transform2base(self,T_marker2cam, T_cam2gripper, T_gripper2base):
        """
        chain rule to get the transformation from marker frame 
        to the robot base frame
        """

        T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam
        
        return T_marker2base


if __name__ == "__main__":
    imgs = sorted(glob.glob('scripts/images/*.png'))
    reproject = Reproject()
    for i in range(len(imgs)):
        reproject.reproject_corners(i)