import cv2
import numpy as np

class aruco_board_detect():
    def __init__(self):

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.arucoParams = cv2.aruco.DetectorParameters()

        self._dist = np.load("scripts/Hand_eye_calibration/distortion_coefficients.npy")
        self._mtx = np.load("scripts/Hand_eye_calibration/camera_matrix.npy")

        self.board_0_3 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                                             dictionary=self.dictionary, ids=np.arange(0,4))
        self.board_4_7 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                                             dictionary=self.dictionary, ids=np.arange(4,8))
        self.board_8_11 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                                              dictionary=self.dictionary, ids=np.arange(8,12))
        self.board_12_15 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                                               dictionary=self.dictionary, ids=np.arange(12,16))
        self.board_16_19 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                                               dictionary=self.dictionary, ids=np.arange(16,20))
        self.board_20_23 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                                               dictionary=self.dictionary, ids=np.arange(20,24))
    
        self._Board = [self.board_0_3, self.board_4_7, self.board_8_11, 
                  self.board_12_15, self.board_16_19, self.board_20_23]

        self.id_0_3 = []
        self.id_4_7 = []
        self.id_8_11 = []
        self.id_12_15 = []
        self.id_16_19 = []
        self.id_20_23 = []

        self.corner_0_3 = []
        self.corner_4_7 = []
        self.corner_8_11 = []
        self.corner_12_15 = []
        self.corner_16_19 = []
        self.corner_20_23 = []
    
        self.axis = np.float32([[0,0,0], [0,0.026,0], [0.026,0.026,0], [0.026,0,0],
                           [0,0,0.026], [0,0.026,0.026], [0.026,0.026,0.026], [0.026,0,0.026]])

    def detect(self, image):

        if True:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            corners, ids, _ = cv2.aruco.detectMarkers(img_gray, 
                                                      dictionary=self.dictionary, 
                                                      parameters=self.arucoParams)
        
            if ids is not None:
                # sort the marker id in ascending order

                switch = np.zeros(6)
                for i in range(len(ids)):
                    _id = int(np.squeeze(ids[i]))
                    if (_id<=3):
                        self.id_0_3.append(_id)
                        self.corner_0_3.append(corners[i])
                        switch[0] = 1
                    elif _id<=7:
                        self.id_4_7.append(_id)
                        self.corner_4_7.append(corners[i])
                        switch[1] = 1
                    elif _id<=11:
                        self.id_8_11.append(_id)
                        self.corner_8_11.append(corners[i])
                        switch[2] = 1
                    elif _id<=15:
                        self.id_12_15.append(_id)
                        self.corner_12_15.append(corners[i])
                        switch[3] = 1
                    elif _id>=16 and _id<=19:
                        self.id_16_19.append(_id)
                        self.corner_16_19.append(corners[i])
                        switch[4] = 1
                    elif _id<=23:
                        self.id_20_23.append(_id)
                        self.corner_20_23.append(corners[i])
                        switch[5] = 1
                    else:
                        print("marker index not recognised")
                        break

                total_id = (np.array(self.id_0_3), np.array(self.id_4_7), np.array(self.id_8_11), 
                                np.array(self.id_12_15), np.array(self.id_16_19), np.array(self.id_20_23))
                sorted_corners = (self.corner_0_3, self.corner_4_7, self.corner_8_11, 
                                  self.corner_12_15, self.corner_16_19, self.corner_20_23)
                    
                for j in range(len(switch)):
                    if switch[j] == 1:
                        _r, _t = self.markers_pose(image, self._Board[j], total_id[j], 
                                                       sorted_corners[j], self._mtx, 
                                                       self._dist, self.axis)
                        break
                
                return _r, _t 
            return None, None

    def markers_pose(self, img, board, ids, corners, _mtx, _dist, axis):
        cv2.aruco.drawDetectedMarkers(img, corners)
        obj_pt, img_pt = board.matchImagePoints(corners, ids)

        _, _rvec, _tvec = cv2.solvePnP(obj_pt, img_pt, _mtx, _dist)
        _rvec, _tvec = cv2.solvePnPRefineVVS(obj_pt, img_pt, _mtx, _dist, _rvec, _tvec)

        markersOfBoardDetected = int(len(obj_pt) / 4)
        if markersOfBoardDetected:
            cv2.drawFrameAxes(img, _mtx, _dist, _rvec, _tvec, 0.01)
            cv2.imwrite("scripts/markers_detected/marker_pose/image_test.png", img)
            # project the cube to the image plane
            # imgpts, _ = cv2.projectPoints(axis, _rvec, _tvec, _mtx, _dist)
            # img = self.draw_3d_cube(img,imgpts)
    
        return _rvec, _tvec

    def draw_3d_cube(self, img, points):
        imgpts = np.int32(points).reshape(-1,2)
        for i,j in zip(range(4),range(4,8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
        img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,),3)

        return img


if __name__ == "__main__":
    aruco_board = aruco_board_detect()