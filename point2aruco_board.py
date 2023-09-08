import numpy as np
import time, glob, cv2
from pymycobot import MyCobotSocket
from pymycobot import PI_PORT, PI_BAUD
from single_aruco_detection import marker_detecting
from forward_kinematics import F_K
from inverse_kinematics import I_K
from scripts.aruco_board_detection import aruco_board_detect


class point2marker():
    def __init__(self):
        #####################CONNECTION##########################
        self.robot = MyCobotSocket("172.25.128.1",9000)
        self.robot.connect("/dev/ttyAMA0","1000000")
        self.robot.power_on()
        #####################CONNECTION##########################

        # load the camera intrinsis matrix and distortion matrix
        self.mtx = np.load("scripts/Hand_eye_calibration/camera_matrix.npy")
        self.dist = np.load("scripts/Hand_eye_calibration/distortion_coefficients.npy")

        # load T_cam2ee from the result of hand-eye calibration
        self.T_cam2ee = np.load("scripts/Hand_eye_calibration/cam2gripper.npy")

        # adjust the pose to detect the target
        self.robot.sync_send_angles([0,0,0,0,0,0], 30)

        self.init_coord = [200,0,250,180,0,-90]


    def run(self):
        # move to the defined pose
        self.robot.sync_send_coords(self.init_coord, 30, mode=1)

        # define the camera
        cap = cv2.VideoCapture(2)

        while cap.isOpened():
            success,img = cap.read()
            if success:
                # using aruco_board
                ar_board = aruco_board_detect()
                rvecs, tvecs = ar_board.detect(img)
                if rvecs is not None:
                    # convert the rvecs and tvecs to homo transformation matrix
                    T_marker2cam = np.eye(4)
                    T_marker2cam[0:3,0:3] = cv2.Rodrigues(rvecs)[0]
                    T_marker2cam[0:3,3] = np.array(tvecs).reshape(3,)

                    # calculate the T_ee2base 
                    ik = I_K()
                    joint_angles = self.robot.get_angles()
                    # forward kinematics T_ee2base
                    T_ee2base = ik.forward(joint_angles)

                    # calculate the T_marker2base
                    T_marker2base = T_ee2base @ self.T_cam2ee @ T_marker2cam

                    # move
                    coord = [200,0,250,180,0,-90]
                    coord[0] = np.round(T_marker2base[0,3]*1000, decimals=0)
                    coord[1] = np.round(T_marker2base[1,3]*1000, decimals=0)

                    print(coord)
                    print()

                    self.robot.sync_send_coords(coord, 30, mode=0)
                    
                    cv2.waitKey(200)



if __name__ == "__main__":
    point = point2marker()
    point.run()
