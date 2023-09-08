from pymycobot import MyCobotSocket
from pymycobot import PI_PORT, PI_BAUD
import time
import os
import cv2
import glob
import math
import numpy as np
from camera_calibration import Camera_Calibration
from forward_kinematics import F_K

class hand_eye_calibration():
    def __init__(self):
        #####################CONNECTION##########################
        self.robot = MyCobotSocket("172.25.128.1",9000)
        self.robot.connect("/dev/ttyAMA0","1000000")
        self.robot.power_on()
        #####################CONNECTION##########################

        # create the dictionary to store the image for calibration
        self._path = os.getcwd() + "/scripts/images"
        if not os.path.exists(self._path):
            os.mkdir(self._path)

        self.coords = []
        self.poses = []

    def get_chess_board(self):
        """
        manully capture the image and joint angles without defining the poses
        """
        # determind the camera
        self.cap = cv2.VideoCapture(2)
        # release the servos
        self.robot.release_all_servos()
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
                    self.coords.append(self.robot.get_coords())
                    cv2.imwrite(self._path + '/img' + str(num) + '.png', img)
                    print(num+1,"image saved!")
                    num += 1
                    if num == 8:
                        break
        cv2.destroyAllWindows()


    def get_chess_board_auto(self):
        """
        automatically capture the image and joint angles 
        please define the poses beforhand
        """
        _pose = (np.load("scripts/Hand_eye_calibration/angles_defined.npy"))
        num = 0
        for i in range(len(_pose)):
            self.robot.sync_send_angles(_pose[i], 30)
            if num < len(_pose):
                self.capture_chess_board(num)
                num += 1
        cv2.destroyAllWindows()

    def capture_chess_board(self, num):
        # determind the camera
        self.cap = cv2.VideoCapture(2)

        while self.cap.isOpened():
            success, img = self.cap.read()
            if success:
                cv2.imwrite(self._path + '/img' + str(num) + '.png', img)
                # read the joints angles
                self.coords.append(self.robot.get_angles())
                self.poses.append(self.robot.get_coords())
                cv2.imshow("img", img)
                cv2.waitKey(100)
                print(num+1,"image saved!")
                num += 1
                # wait 1s
                break

        self.cap.release()
        cv2.destroyAllWindows()

def translation_inverse(r,t):
    """
    calculate the translation part for the inverse matrix of homogeneous transformation matrix
    """
    mat = []
    for i in range(len(t)):
        _t = (-1) * np.dot(r[i],(t[i]))
        mat.append(_t)
    return mat

def inverse(rvec, tvec):
    mat = []
    for i in range(len(tvec)):
        eye_mat = np.eye(4)
        _r,_ = cv2.Rodrigues(rvec[i])
        eye_mat[0:3,0:3] = np.array(_r).reshape(3,3)
        eye_mat[0:3,3] = np.array(tvec[i]).reshape(3,)
        mat.append(eye_mat)
    
    t = []
    r = []
    for j in range(len(tvec)):
        _inv_T = np.linalg.inv(mat[j])
        r.append(_inv_T[0:3,0:3])
        t.append(_inv_T[0:3,3])
    
    return r,t


def vector2matrix(rvec, tvec, flag):
    """
    convert the rotation vector and translation vector into homogenous transformation matrix
    """
    if flag < 0:
    # inverse the transformation
        rotmat = []
        for _rvec in rvec:
            _r, _ = cv2.Rodrigues(_rvec)
            # transpose the rotation matrix
            _r_t = np.array(_r).T
            rotmat.append(_r_t)
            # print()
        _tvec = translation_inverse(rotmat, tvec)
        return np.array(rotmat), np.array(_tvec)
    
    else:
        rotmat = []
        for _rvec in rvec:
            _r, _ = cv2.Rodrigues(_rvec)
            rotmat.append(_r)
            # print()
        
        return np.array(rotmat), np.array(tvec)

# # Checks if a matrix is a valid rotation matrix.
# def isRotationMatrix(R) :
#     Rt = np.transpose(R)
#     shouldBeIdentity = np.dot(Rt, R)
#     I = np.identity(3, dtype = R.dtype)
#     n = np.linalg.norm(I - shouldBeIdentity)
#     return n < 1e-6
 
# # Calculates rotation matrix to euler angles
# # The result is the same as MATLAB except the order
# # of the euler angles ( x and z are swapped ).
# def rotationMatrixToEulerAngles(R) :

#     assert(isRotationMatrix(R))
 
#     sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
#     singular = sy < 1e-6
 
#     if  not singular :
#         x = math.atan2(R[2,1] , R[2,2])
#         y = math.atan2(-R[2,0], sy)
#         z = math.atan2(R[1,0], R[0,0])
#     else :
#         x = math.atan2(-R[1,2], R[1,1])
#         y = math.atan2(-R[2,0], sy)
#         z = 0
 
#     return np.array([x, y, z])



def main():
    calibration = hand_eye_calibration()
    calibration.robot.sync_send_angles([0,0,0,0,0,0], 20)
    print(calibration.robot.get_coords())
    time.sleep(0.5)

    #########################################################################

    # # get the chess board imaages and record the pose of the robot of each image
    calibration.get_chess_board_auto()

    # save the joint angles
    np.save("scripts/Hand_eye_calibration/angles_current", calibration.coords)

    #########################################################################
    # debug
    # load the recorded angles
    # coords = np.load("scripts/Hand_eye_calibration/angles_current.npy")

    #########################################################################

    # T_target2cam
    camera = Camera_Calibration()

    # calibrate the camera to get the intrinsic matrix and distortion matrix
    camera.camera_calibration(mode="cam")

    # load the known camera matrix and distortion coefficient to get the transformation matrix
    camera.camera_calibration(mode="hand_eye")
    # from the solvePnp function, the rvecs and tvecs is the T from world to camera which can be treated as from the target to camera
    R_target2cam, t_target2cam = vector2matrix(camera.rvecs, camera.tvecs, flag=1)

    np.save("scripts/Hand_eye_calibration/R_target2cam", R_target2cam)
    np.save("scripts/Hand_eye_calibration/t_target2cam", t_target2cam)

    #########################################################################
    # T_gripper2base

    fk = F_K()
    # use the forward kinematics to calculate the transformation matrix

    ##### replace "calibration.coords" to "coords" when debuging #####
    for i in range(len(calibration.coords)):
        fk.forward(calibration.coords[i], num=i, flag="handeye")
    
    T_gripper2base_file = sorted(glob.glob("scripts/Hand_eye_calibration/T_gripper2base/*.npy"))

    # for item in T_gripper2base_file:
    #     print(item)

    T_gripper2base = [np.load(f) for f in T_gripper2base_file]

    R_gripper2base = [t[0:3,0:3] for t in T_gripper2base]
    t_gripper2base = [t[0:3,-1] for t in T_gripper2base]

    # for item in T_gripper2base:
    #     print(item)
    #     print()

    # r_file = sorted(glob.glob("scripts/Hand_eye_calibration/R_gripper2base/*.npy"))
    # t_file = sorted(glob.glob("scripts/Hand_eye_calibration/t_gripper2base/*.npy"))
    # R_gripper2base = [np.load(f) for f in r_file]
    # # convert to rotation vectors
    # R_gripper2base = [cv2.Rodrigues(r)[0] for r in R_gripper2base]
    # t_gripper2base = [np.load(f) for f in t_file]

    np.save("scripts/Hand_eye_calibration/R_gripper2base", R_gripper2base)
    np.save("scripts/Hand_eye_calibration/t_gripper2base", t_gripper2base)
    
    #########################################################################
    # Hand-eye calibration
    """
    cv::HandEyeCalibrationMethod {
    cv::CALIB_HAND_EYE_TSAI = 0,
    cv::CALIB_HAND_EYE_PARK = 1,
    cv::CALIB_HAND_EYE_HORAUD = 2,
    cv::CALIB_HAND_EYE_ANDREFF = 3,
    cv::CALIB_HAND_EYE_DANIILIDIS = 4
    }
    """

    R_cam2gripper, t_cam2gripper =cv2.calibrateHandEye(R_gripper2base, t_gripper2base, 
                                                       R_target2cam, t_target2cam, 
                                                       method=4)
    
    _T = np.eye(4,4)
    _T[0:3,0:3] = R_cam2gripper
    _T[0:3,-1] = np.reshape(t_cam2gripper, (3,))

    print(_T)
    print()

    np.save("scripts/Hand_eye_calibration/cam2gripper", _T)


if __name__ == "__main__":
    main()
