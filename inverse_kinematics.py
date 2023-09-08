import numpy as np
import math, cv2

class I_K():
    def __init__(self):
        # "T" is the transformation matrix

        # define the learning rate
        self.alpha = 0.1

        # counter
        self.count = 0
        self.max_count = 5000


    def inverse(self, T_target, initial_joints):
        # "initial_joints" is in degree

        # initial joint angles
        joints_prev = (initial_joints).copy()

        # convert the transformation matrix to configuration space
        q_target = self.T2configuration(T_target)
        
        while True:
            # calculate the transformation matrix of last joints space
            T_prev = self.forward(joints_prev, len(joints_prev))

            # delta is a (6,1) vector representing the difference of configuration space between
            # the initial guess and the target
            delta = q_target - self.T2configuration(T_prev)

            # joints_current is in radian
            joints_current = np.deg2rad(np.array(joints_prev).reshape(6,1)) + self.alpha * (self.get_jacobian(joints_prev).T @ delta)

            error = np.linalg.norm(self.get_jacobian(joints_prev).T @ delta)
            if (error < 1e-02) or (self.count >= self.max_count):
                print("the error is", error, end="\n")
                print("the interaction is", self.count, end="\n")
                break

            joints_prev = joints_current.copy() # in radian
            joints_prev = np.rad2deg(joints_prev) # to degree

            self.count += 1
        
        return np.rad2deg(joints_current), error


    def T2configuration(self, T):
        q = np.zeros((6,1))
        q[0:3,0] = np.array(T[0:3,3]).reshape((3,))
        q[3:,0] = np.array(cv2.Rodrigues(T[0:3,0:3])[0]).reshape(3,)

        return q



    def get_jacobian(self, joints_angles):
        # "joints_angles" is in degree

        # calculate the jacobian matrix
        # there are 6 joints in the elephant robot, so the dimension of the 
        # jacobian matrix is (6,6)

        # calculate the transformation matrix to the final joint
        P = self.forward(joints_angles, len(joints_angles))[0:3,3]

        # intialise the jacobian matrix
        jacobian = np.ones((6,6))

        P0 = np.zeros((3))
        Z0 = np.array([[0,0,1]])

        for i in range(len(joints_angles)):
            T = self.forward(joints_angles, i)
            jacobian[0:3, i] = np.cross(T[0:3,2], (P - T[0:3,3]))
            jacobian[3:, i] = T[0:3,2]

        jacobian[0:3,0]=np.cross(Z0,(P-P0))
        jacobian[3:,0]=Z0[0].tolist()

        return jacobian



    def forward(self, joints_angles, up_to_joint=6):
        # "joints_angles" is in degree

        # "up_to_joint" represent the number of joint, in this case, it should be [1~6]

        # define the DH parameters according to the document of the robot
        dh_parameters = {
            "a": [0.0, -0.1104, -0.0960, 0.0, 0.0, 0.0],
            "alpha": [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0],
            "d": [0.13122, 0.0, 0.0, 0.0634, 0.07505, 0.0456],
            #   "theta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "offset": [0.0, -np.pi / 2, 0.0, -np.pi / 2, np.pi / 2, 0.0]
        }

        # angles are in degrees
        joint_reading = joints_angles

        T = np.eye(4)

        # calculate the transformation to specific joint
        for i in range(up_to_joint):
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

        return T


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
    ik = I_K()
    initial_joints = [0,0,0,0,0,0]
    T_target = np.load("scripts/marker_tracking/T_base2marker.npy")
    print("T_target", T_target, end="\n")
    q,error = ik.inverse(T_target, initial_joints)

    print(q, end="\n")