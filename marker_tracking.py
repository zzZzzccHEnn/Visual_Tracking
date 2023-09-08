import numpy as np
import time, glob, cv2
from pymycobot import MyCobotSocket
from pymycobot import PI_PORT, PI_BAUD
from forward_kinematics import F_K
from inverse_kinematics import I_K
from aruco_board_detection import aruco_board_detect



class aruco_board_tracking():
    def __init__(self):
        #####################CONNECTION##########################
        self.robot = MyCobotSocket("172.25.128.1",9000)
        self.robot.connect("/dev/ttyAMA0","1000000")
        self.robot.power_on()
        #####################CONNECTION##########################

        # load the camera intrinsis matrix and distortion matrix
        self.mtx = np.load("scripts/Hand_eye_calibration"
                           + "/camera_matrix.npy")
        self.dist = np.load("scripts/Hand_eye_calibration"
                            + "/distortion_coefficients.npy")

        # load T_cam2ee from the result of hand-eye calibration
        self.T_cam2ee = np.load("scripts/Hand_eye_calibration"
                                + "/cam2gripper.npy")

        # initial pose
        self.robot.sync_send_angles([0,0,0,0,0,0], 20)

        # define the dictionary and parameters
        self.dictionary = cv2.aruco.getPredefinedDictionary(
                                    cv2.aruco.DICT_6X6_250)
        
        self.arucoParams = cv2.aruco.DetectorParameters()

        # define the object_points
        # the unit of square_length should be "m"
        # for single aruco marker only
        self.square_length = 0.026
        self.obj_points = np.array([
                                [-self.square_length/2, self.square_length/2, 0],
                                [self.square_length/2, self.square_length/2, 0],
                                [self.square_length/2, -self.square_length/2, 0],
                                [-self.square_length/2, -self.square_length/2, 0]
                                ])
        
        # adjust the pose to detect the target
        self.init_coord = [200,0,250,180,0,-90]

        # record the previous coordinate
        self.coord_pre = self.init_coord.copy()

        # record the previous corner pixel coorndinates
        self.pre_corners = np.zeros((4,1,4,2),dtype=float)


    def run(self):
        print("########## START ##########")

        # move to the defined pose
        self.robot.sync_send_coords(self.init_coord, 10, mode=1)

        # define the camera
        cap = cv2.VideoCapture(2)

        ###### place the marker that the orientation 
        # of marker is the same as the robot base ######

        while cap.isOpened():
            success, img = cap.read()
            if success:
                # show the image
                # cv2.imshow("camera", img)
                # cv2.waitKey(1)

                # convert to gray scale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # detect the marker
                (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, 
                                                                self.dictionary, 
                                                                parameters=self.
                                                                arucoParams)
                
                # calculate the change of pixel coordinate of markers
                # as using the aruco board, sometimes 
                # only partial markers is detected
                if np.shape(corners) == np.shape(self.pre_corners):
                    # calculate the average moving distance of each corner
                    corner_diff = np.sqrt(
                                  np.sum(
                                  np.square(
                                  np.array(corners) 
                                  - np.array(self.pre_corners)))) / (4*4)
                    # print(corner_diff)
                else:
                    # no marker detected
                    continue
                
                # if markers detected and the distance 
                # is larger than threshold then
                if len(corners)>0 and corner_diff > 2.0:
                    
                    # store the previous corners pixel coordinate
                    self.pre_corners = corners

                    #################################################################
                    # calculate the T_ee2base 
                    ik = I_K()
                    joint_angles = self.robot.get_angles()
                    # forward kinematics T_ee2base
                    T_ee2base = ik.forward(joint_angles)

                    #################################################################
                    # # calculate the T_marker2cam (single aruco marker)
                    # _, rvecs, tvecs = cv2.solvePnP(self.obj_points, 
                    #                                corners[0], 
                    #                                self.mtx, self.dist, 
                    #                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    
                    #################################################################
                    # calculate the T_marker2cam (aruco_board)
                    ar_board = aruco_board_detect()
                    rvecs, tvecs = ar_board.detect(img)

                    #################################################################
                    # convert the rvecs and tvecs to homo transformation matrix
                    T_marker2cam = np.eye(4)
                    T_marker2cam[0:3,0:3] = cv2.Rodrigues(rvecs)[0]
                    T_marker2cam[0:3,3] = np.array(tvecs).reshape(3,)

                    #################################################################
                    # calculate the T_marker2base
                    T_marker2base = T_ee2base @ self.T_cam2ee @ T_marker2cam

                    #################################################################
                    # move
                    coord = [200,0,250,180,0,-90]
                    coord[0] = np.round(T_marker2base[0,3]*1000, decimals=-1)
                    coord[1] = np.round(T_marker2base[1,3]*1000, decimals=-1)

                    # print(coord)
                    # print()

                    # small step for each movement
                    tiny_step = 5
                    if coord[0]-self.coord_pre[0] > tiny_step:
                        coord[0] = self.coord_pre[0] + tiny_step
                    elif coord[0]-self.coord_pre[0] < - tiny_step:
                        coord[0] = self.coord_pre[0] - tiny_step
                    
                    if coord[1]-self.coord_pre[1] > tiny_step:
                        coord[1] = self.coord_pre[1] + tiny_step
                    elif coord[1]-self.coord_pre[1] < - tiny_step:
                        coord[1] = self.coord_pre[1] - tiny_step


                    self.robot.send_coords(coord, 10, mode=0)
                    self.coord_pre = coord.copy()

                    cv2.imshow("camera", img)
                    # change the number in "waitKey()" for different "fps"
                    cv2.waitKey(1)

if __name__ == "__main__":
    tracking = aruco_board_tracking()
    tracking.run()

