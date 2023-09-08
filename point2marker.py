import numpy as np
import time, glob, cv2
from pymycobot import MyCobotSocket
from pymycobot import PI_PORT, PI_BAUD
from single_aruco_detection import marker_detecting
from forward_kinematics import F_K
from inverse_kinematics import I_K


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

        self.joints_angle_current = []

        self.angles_defined = np.load("scripts/marker_tracking/angles_defined.npy")

        # adjust the pose to detect the target
        self.robot.sync_send_angles([0,0,0,0,0,0], 50)


    def run(self):
        print("########## START ##########")

        # load T_cam2gripper from the result of hand-eye calibration
        T_cam2gripper = np.load("scripts/Hand_eye_calibration/cam2gripper.npy")

        # move to the defined poses
        for i in range(len(self.angles_defined)):
            self.robot.sync_send_angles(self.angles_defined[i], 50)
            self.capture_image(num=i)
            time.sleep(0.5)

        np.save("scripts/marker_tracking/joints_angle_current", self.joints_angle_current)

        self.robot.sync_send_angles([0,0,0,0,0,0], 50)

        #########################################################################
        # EE2Base/Gripper2Base

        current_joints = np.load("scripts/marker_tracking/joints_angle_current.npy")
        # use the forward kinematics to calculate the transformation
        # between ee and base
        fk = F_K()
        for i in range(len(current_joints)):
            fk.forward(current_joints[i], num=i, flag="tracking")
        
        # load the T of base 2 ee
        T_ee2base_files = sorted(glob.glob("scripts/marker_tracking/T_gripper2base_tracking/*.npy"))
        T_ee2base = [np.load(f) for f in T_ee2base_files]

        #########################################################################
        # Marker2Cam

        # load the images files
        marker_images = sorted(glob.glob("scripts/markers_detected/*.png"))

        T_marker2cam = []
        
        # detect the marker then calculate the rvecs and tvecs
        for i in range(len(marker_images)):
            image_file = marker_images[i]
            image = cv2.imread(image_file)
            rvecs, tvecs = self.marker_pose(image, i)

            T = np.eye(4)
            T[0:3,0:3] = cv2.Rodrigues(rvecs)[0]
            T[0:3,3] = np.array(tvecs).reshape(3,)
            T_marker2cam.append(T)

        #########################################################################
        # transform the target to base frame

        T_marker2base = self.transform2base(T_marker2cam, T_cam2gripper, T_ee2base)

        coord = [0,0,20,180,0,-90]

        coord[0] = T_marker2base[0,3]*1000
        coord[1] = T_marker2base[1,3]*1000


        #########################################################################
        # Move to the marker
        print(coord)
        print()

        self.robot.send_coords(coord, 20, mode=1)


    #########################################################################

    def transform2base(self, T_marker2cam, T_cam2gripper, T_gripper2base):
        """
        chain rule to get the transformation from marker frame to the robot base frame
        """
        assert (np.array(T_marker2cam).size == np.array(T_gripper2base).size), "the sizes of T_marker2cam and T_gripper2base are not the same"

        for i in range(1):
            # the result is the transformation from marker to base
            T_marker2base = T_gripper2base[i] @ T_cam2gripper @ T_marker2cam[i]
            

        np.save("scripts/marker_tracking/T_marker2base", T_marker2base)
        
        return T_marker2base



    def marker_pose(self, img, num):
        """
        estimate the marker pose in the image
        """
        # define the dictionary and parameters
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        arucoParams = cv2.aruco.DetectorParameters()

        # convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect the marker
        (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, dictionary, 
                                                           parameters=arucoParams)
        
        # define the object_points
        # the unit of square_length should be "m"
        square_length = 0.026
        obj_points = np.array([[-square_length/2, square_length/2, 0],
                               [square_length/2, square_length/2, 0],
                               [square_length/2, -square_length/2, 0],
                               [-square_length/2, -square_length/2, 0]])

        if len(corners)>0:
            for i in range(len(ids)):
                # get the rotation and translation vectors
                _, rvecs, tvecs = cv2.solvePnP(obj_points, corners[i], 
                                                self.mtx, self.dist, 
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
                
                cv2.drawFrameAxes(img, self.mtx, self.dist, rvecs, tvecs, square_length)
                _path = "scripts/markers_detected/marker_pose/img_" + str(num) + ".png"
                cv2.imwrite(_path, img)
            return rvecs, tvecs



    def capture_image(self, num):
        cap = cv2.VideoCapture(2)

        while cap.isOpened():
            success, img = cap.read()
            if success:
                file_name = "scripts/markers_detected/img_" + str(num) + ".png"
                # capture the image
                cv2.imwrite(file_name, img)
                # store the joints angles
                self.joints_angle_current.append(self.robot.get_angles())
                break
        cap.release()              

    

if __name__ == "__main__":
    marker_pose = point2marker()
    marker_pose.run()

