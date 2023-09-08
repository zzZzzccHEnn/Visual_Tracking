from pymycobot import MyCobotSocket
from pymycobot import PI_PORT, PI_BAUD
import time
import numpy as np
import cv2
from forward_kinematics import F_K
from pymycobot.genre import Coord
from aruco_board_detection import aruco_board_detect


def main():
    robot = MyCobotSocket("172.25.128.1",9000)
    robot.connect("/dev/ttyAMA0","1000000")
    robot.power_on()

    # robot.release_all_servos()
    robot.sync_send_angles([0,0,0,0,0,0], 50)
    # robot.sync_send_coords([200,0,250,180,0,-90], 30, mode=1)

# def cam():
#     cap = cv2.VideoCapture(2)
#     while cap.isOpened():
#         success, img = cap.read()
#         if success:
#             D = aruco_board_detect()
#             _rvec, _tvec = D.detect(img)

#             print(_rvec)
#             print(_tvec)
#             print()
#             cv2.waitKey(100)


if __name__ == "__main__":
    main()
    # cam()