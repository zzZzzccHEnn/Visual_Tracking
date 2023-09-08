import numpy as np
import time, glob, cv2
from pymycobot import MyCobotSocket
from pymycobot import PI_PORT, PI_BAUD
from single_aruco_detection import marker_detecting
from forward_kinematics import F_K
from inverse_kinematics import I_K
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes


def std_deviation():
    # load the T_cam2ee
    T_cam2ee = np.load("scripts/Hand_eye_calibration/cam2gripper.npy")

    # load the T_marker2cam
    R_target2cam = np.load("scripts/Hand_eye_calibration/R_target2cam.npy")
    t_target2cam = np.load("scripts/Hand_eye_calibration/t_target2cam.npy")

    # number of images
    n = len(R_target2cam)

    T_mar2cam = []
    for i in range(len(t_target2cam)):
        T_eye = np.eye(4)
        T_eye[0:3,0:3] = R_target2cam[i]
        T_eye[0:3,3] = np.reshape(t_target2cam[i], (3,))
        T_mar2cam.append(T_eye)

    # load the T_ee2base
    T_ee2base_file = sorted(glob.glob("scripts/Hand_eye_calibration"
                                      + "/T_gripper2base/*.npy"))
    
    T_ee2base = [np.load(f) for f in T_ee2base_file]

    # T_marker2base
    T_mar2base = []
    for i in range(len(T_ee2base)):
        T_mar2base.append(transform2base(T_mar2cam[i], T_cam2ee, T_ee2base[i]))

    np.save("scripts/Analysis/T_mar2base", T_mar2base)
    # print(np.shape(T_mar2base))
    
    # create the chessboard corners
    # 9X6 chessboard, 0.016m for the edge of the square
    chessboard_corners = []
    # for line (y)
    for i in range(0,6):
        # for coloum (x)
        for j in range(0,9):
            corner_coord = [0,0,0,1]
            corner_coord[0] = 0.016 * j
            corner_coord[1] = 0.016 * i
            chessboard_corners.append(corner_coord)

    chessboard_corners = np.reshape(chessboard_corners, (54,4))

    # print(np.shape(chessboard_corners)[0])
    # print(chessboard_corners)


    # transfer the chessboard corners to the base
    y = []
    # loop the T_mar2base
    for i in range(0, np.shape(T_mar2base)[0]):
        # loop the chessboard corners
        y_i = []
        for j in range(0,54):
            coord = T_mar2base[i] @ np.reshape(chessboard_corners[j], (4,1))
            y_i.append(coord)
        y.append(y_i)

    # print(np.shape(y[0]))

    # sum the y_i
    sum = np.squeeze(y[0])
    for i in range(1,n):
        sum = sum + np.squeeze(y[i])

    # y bar
    y_bar = (sum)/n

    # y_i - y_bar
    error = []
    for i in range(0, np.shape(y)[0]):
        error.append(np.squeeze(y[i])-y_bar)

    # print(np.shape(error[0]))

    # square each error, then sum, then divided by 6*9, finally square root
    # error_each_image in m
    # error of each corresponding corner in each image
    error_each_image = []
    for i in range(0, n):
        error_each_image.append(np.sqrt(np.sum(np.square(error[i])))/(6*9))

    # print(error_each_image)


    ##### draw the figure and save #####
    # font1 = {'family':'times','size':14}
    # font2 = {'family':'times','size':12}
    # plt.figure(figsize=(12,5))
    # plt.plot(np.arange(1,33), error_each_image)
    # plt.xticks(range(1,33))
    # plt.xlabel("No.i pose", fontdict=font2)
    # plt.ylabel("Error / (m)", fontdict=font2)
    # plt.title("Average error at No.i pose", fontdict=font1)
    # plt.savefig("error.png",dpi=500)


    ### procrustes ###
    R, _ = orthogonal_procrustes(y_bar, chessboard_corners)
    np.save("scripts/Analysis/T_mar2base_procrustes", R)

    print("the estimated transformation is:")
    print(R)


def transform2base(T_marker2cam, T_cam2gripper, T_gripper2base):
    """
    chain rule to get the transformation from marker frame to the robot base frame
    """

    T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam
        
    return T_marker2base


if __name__ == "__main__":
    std_deviation()
    