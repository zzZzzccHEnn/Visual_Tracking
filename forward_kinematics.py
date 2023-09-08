import numpy as np
import math

class F_K():

    def forward(self, joints, num = None, flag=None):
        # define the DH parameters according to the document of the robot
        dh_parameters = {
            "a": [0.0, -0.1104, -0.0960, 0.0, 0.0, 0.0],
            "alpha": [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0],
            "d": [0.13122, 0.0, 0.0, 0.0634, 0.07505, 0.0456],
            #   "theta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "offset": [0.0, -np.pi / 2, 0.0, -np.pi / 2, np.pi / 2, 0.0]
        }

        # angles are in degrees
        joint_reading = joints

        T = np.eye(4)
        for i in range(len(dh_parameters["a"])):
            #    retrieve the dh parameters
            a = dh_parameters["a"][i]
            alpha = dh_parameters["alpha"][i]
            d = dh_parameters["d"][i]
            offset = dh_parameters["offset"][i]

            # convert the joints angle from degree to radian
            theta = math.radians(joint_reading[i]) + offset

            #    calculate the transform matrix
            T_i = self.transfer(a, alpha, d, theta)

            T = T.dot(T_i)

            if flag == "handeye":
                filename = "scripts/Hand_eye_calibration/T_gripper2base/T_gripper2base" + str(num)
                np.save(filename, T)
            
            if flag == "tracking":
                filename = "scripts/marker_tracking/T_gripper2base_tracking/T_gripper2base" + str(num)
                np.save(filename, T)


    def transfer(self, a, alpha, d, theta):
        A = np.zeros((4, 4))

        A[0, 0] = np.cos(theta)
        A[0, 1] = -np.sin(theta) * np.cos(alpha)
        A[0, 2] = np.sin(theta) * np.sin(alpha)
        A[0, 3] = a * np.cos(theta)
        A[1, 0] = np.sin(theta)
        A[1, 1] = np.cos(theta) * np.cos(alpha)
        A[1, 2] = -np.cos(theta) * np.sin(alpha)
        A[1, 3] = a * np.sin(theta)
        A[2, 1] = np.sin(alpha)
        A[2, 2] = np.cos(alpha)
        A[2, 3] = d
        A[3, 3] = 1.0

        return A


if __name__ == "__main__":
    # load the joints angles, the angles are in degrees
    angles = np.load("scripts/angles_current.npy")
    fk = F_K()
    
    for i in range(len(angles)):
        fk.forward(joints=angles[i], num=i)



