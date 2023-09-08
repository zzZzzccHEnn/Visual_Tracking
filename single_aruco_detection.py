import cv2
import numpy as np
from camera_calibration import Camera_Calibration
import matplotlib.pyplot as plt

# cv2.__version__ = 4.8
class marker_detecting():
    def __init__(self):
        # dertermind the camera
        self.cap = cv2.VideoCapture(2)

        # load the calibration result
        self._dist = np.load("scripts/Hand_eye_calibration/distortion_coefficients.npy")
        self._mtx = np.load("scripts/Hand_eye_calibration/camera_matrix.npy")

        # start
        self.detecting(self._mtx, self._dist)


    def detecting(self, mtx, dist):
        # determind the aruco marker dictionary
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        cv2.namedWindow("Detection", 0)

        while True:
            # read the image from video
            ret, img = self.cap.read()
            # img = cv2.imread("scripts/3.jpg")

            # print(img.shape)
            # (480, 640, 3)

            # convert to gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            arucoParams = cv2.aruco.DetectorParameters()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, dictionary, parameters=arucoParams)
            
            square_length = 0.026
            obj_points = np.array([[-square_length/2, square_length/2, 0],
                                   [square_length/2, square_length/2, 0],
                                   [square_length/2, -square_length/2, 0],
                                   [-square_length/2, -square_length/2, 0]])

            # if marker is detected
            if len(corners) > 0:
                for i in range(len(ids)):
                    # get the rotation and translation vectors
                    _, self.rvecs, self.tvecs = cv2.solvePnP(obj_points, corners[i], mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)

                    # _x = []
                    # _y = []
                    # for i in range(np.shape(corners)[0]):
                    #     _x.append([x for [x, y] in corners[i][0]])
                    #     _y.append([y for [x, y] in corners[i][0]])


                    # top_left_x = np.array([min(x) for x in _x]).astype(int)
                    # top_left_y = np.array([min(y) for y in _y]).astype(int)
                    # bottom_right_x = np.array([max(x) for x in _x]).astype(int)
                    # bottom_right_y = np.array([max(y) for y in _y]).astype(int)
                    
                    # print("_x", _x)
                    # print("_y", _y)
                    # print(top_left_x,top_left_y,bottom_left_x,bottom_left_y)

                    # warp_img = []
                    # for j in range(len(top_left_x)):
                    #     warp_img.append(img[top_left_y[j]:bottom_right_y[j]+1, top_left_x[j]:bottom_right_x[j]+1])

                    # cv2.aruco.drawDetectedMarkers(img, corners, ids) 
                    # cv2.drawFrameAxes(img, mtx, dist, rvecs, tvecs, 0.01)
                    # print("wrap images", len(warp_img))
                    # orb = cv2.ORB().create(5000)
                    # for k in range(len(warp_img)):
                    #     kp = orb.detect(warp_img[k])
                    #     _img = cv2.drawKeypoints(warp_img[k],kp, color=[0,255,0], outImage=None)
                    #     cv2.imshow("marker_"+str(ids[k][0]), _img)

            # cv2.imshow("Detection", img)
            # 20fps "cv2.waitKey(50)"
            # cv2.waitKey(1)     
        

if __name__ == "__main__":
    detect = marker_detecting()