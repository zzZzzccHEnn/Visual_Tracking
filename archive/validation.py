import numpy as np

class validate():
    def kinematics(self):
        P_target = np.array([200,0,0,1]).reshape(4,1)

        R_gripper2base = np.load("scripts/R_gripper2base.npy")
        t_gripper2base = np.load("scripts/t_gripper2base.npy")

        R_target2cam = np.load("scripts/R_target2cam.npy")
        t_target2cam = np.load("scripts/t_target2cam.npy")

        _T = np.load("scripts/cam2gripper.npy")

        H_gripper2fbase = []
        H_target2cam = []

        for i in range(len(R_gripper2base)):
            H_gripper2fbase.append(self.homomat(R_gripper2base[i], t_gripper2base[i]))
            H_target2cam.append(self.homomat(R_target2cam[i], t_target2cam[i]))

        for j in range(len(R_gripper2base)):
            P_base = np.array(H_gripper2fbase[j]).dot(_T.dot(np.array(H_target2cam[j]).dot(P_target)))
            print(np.reshape(P_base,(1,4)))

    def homomat(self, r, t):
        H = np.eye(4,4)
        H[0:3,0:3] = r
        H[0:3,-1] = np.reshape(t, (3,))

        return H

if __name__ == "__main__":
    validation = validate()
    validation.kinematics()
