import numpy as np
import cv2, time
from pymycobot import MyCobotSocket
from pymycobot import PI_PORT, PI_BAUD

class define_coords():
    def __init__(self, _p):
        #####################CONNECTION##########################
        self.robot = MyCobotSocket("172.25.128.1",9000)
        self.robot.connect("/dev/ttyAMA0","1000000")
        self.robot.power_on()
        #####################CONNECTION##########################

        self.cap = cv2.VideoCapture(2)
        self.angles = []

        self.path = _p

    def run(self):
        print("#####Start Capture#####")
        self.robot.sync_send_angles([0,0,0,0,0,0], 50)
        time.sleep(1)

        # determine the number of poses
        n = 32
        self.capture_coords(n)

        print("recording done")

        # save
        np.save(self.path, self.angles)
    
    def capture_coords(self, n):
        num = 0
        self.robot.release_all_servos()

        while self.cap.isOpened():
            ret,img = self.cap.read()
            if ret:
                k = cv2.waitKey(5)
                cv2.imshow("img", img)
                if k == 27:
                    break
                elif k == ord("s"):
                    self.angles.append(self.robot.get_angles())
                    print(self.robot.get_coords())
                    print(num+1, "coordinate recorded")
                    num += 1
                    if num == n:
                        break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # main()
    # _path = "scripts/marker_tracking/angles_defined"
    _path = "scripts/Hand_eye_calibration/angles_defined"
    capture = define_coords(_path)
    capture.run()