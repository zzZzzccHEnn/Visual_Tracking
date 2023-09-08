import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(2)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters()

    _dist = np.load("scripts/Hand_eye_calibration/distortion_coefficients.npy")
    _mtx = np.load("scripts/Hand_eye_calibration/camera_matrix.npy")

    board_0_3 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                            dictionary=dictionary, ids=np.arange(0,4))
    board_4_7 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                            dictionary=dictionary, ids=np.arange(4,8))
    board_8_11 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                            dictionary=dictionary, ids=np.arange(8,12))
    board_12_15 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                            dictionary=dictionary, ids=np.arange(12,16))
    board_16_19 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                            dictionary=dictionary, ids=np.arange(16,20))
    board_20_23 = cv2.aruco.GridBoard(size=(2,2), markerLength=0.0125, markerSeparation=0.001, 
                            dictionary=dictionary, ids=np.arange(20,24))

    
    _Board = [board_0_3, board_4_7, board_8_11, board_12_15, board_16_19, board_20_23]

    id_0_3 = []
    id_4_7 = []
    id_8_11 = []
    id_12_15 = []
    id_16_19 = []
    id_20_23 = []

    corner_0_3 = []
    corner_4_7 = []
    corner_8_11 = []
    corner_12_15 = []
    corner_16_19 = []
    corner_20_23 = []
    
    axis = np.float32([[0,0,0], [0,0.026,0], [0.026,0.026,0], [0.026,0,0],
                       [0,0,0.026], [0,0.026,0.026], [0.026,0.026,0.026], [0.026,0,0.026]])

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            corners, ids, _ = cv2.aruco.detectMarkers(img_gray, dictionary=dictionary, 
                                                      parameters=arucoParams)
        

            if ids is not None:
                # # sort the marker id in ascending order
                # _id_sorted = np.sort([np.squeeze(i) for i in ids])
                # _index_sorted = sorted(range(len(_id_sorted)), key=lambda k : ids[k])

                # # sort the corners with the sorted index
                # corners_sorted = []
                # for i in _index_sorted:
                    # corners_sorted.append(corners[i])

                switch = np.zeros(6)
                for i in range(len(ids)):
                    _id = int(np.squeeze(ids[i]))
                    if (_id<=3):
                        id_0_3.append(_id)
                        corner_0_3.append(corners[i])
                        switch[0] = 1
                    elif _id<=7:
                        id_4_7.append(_id)
                        corner_4_7.append(corners[i])
                        switch[1] = 1
                    elif _id<=11:
                        id_8_11.append(_id)
                        corner_8_11.append(corners[i])
                        switch[2] = 1
                    elif _id<=15:
                        id_12_15.append(_id)
                        corner_12_15.append(corners[i])
                        switch[3] = 1
                    elif _id>=16 and _id<=19:
                        id_16_19.append(_id)
                        corner_16_19.append(corners[i])
                        switch[4] = 1
                    elif _id<=23:
                        id_20_23.append(_id)
                        corner_20_23.append(corners[i])
                        switch[5] = 1
                    else:
                        print("marker index not recognised")
                        break

                total_id = (np.array(id_0_3), np.array(id_4_7), np.array(id_8_11), 
                            np.array(id_12_15), np.array(id_16_19), np.array(id_20_23))
                sorted_corners = (corner_0_3, corner_4_7, corner_8_11, corner_12_15, corner_16_19, corner_20_23)
                    
                for j in range(len(switch)):
                    if switch[j] == 1:
                        img = detect_markers(img, _Board[j], total_id[j], sorted_corners[j], _mtx, _dist, axis)
                
            cv2.imshow("marker", img)
            cv2.waitKey(1000)

            switch = np.zeros(6)

            id_0_3 = []
            id_4_7 = []
            id_8_11 = []
            id_12_15 = []
            id_16_19 = []
            id_20_23 = []

            corner_0_3 = []
            corner_4_7 = []
            corner_8_11 = []
            corner_12_15 = []
            corner_16_19 = []
            corner_20_23 = []   

def detect_markers(img, board, ids, corners, _mtx, _dist, axis):
    cv2.aruco.drawDetectedMarkers(img, corners)
    obj_pt, img_pt = board.matchImagePoints(corners, ids)
    # print("obj_pt",obj_pt)
                # print("img_pt",img_pt)
    _, _rvec, _tvec = cv2.solvePnP(obj_pt, img_pt, _mtx, _dist)
    _rvec, _tvec = cv2.solvePnPRefineVVS(obj_pt, img_pt, _mtx, _dist, _rvec, _tvec)

    markersOfBoardDetected = int(len(obj_pt) / 4)
                # print("markersOfBoardDetected", np.shape(obj_pt))
    if markersOfBoardDetected:
        cv2.drawFrameAxes(img, _mtx, _dist, _rvec, _tvec, 0.01)
        # project the cube to the image plane
        imgpts, _ = cv2.projectPoints(axis, _rvec, _tvec, _mtx, _dist)
        img = draw_3d_cube(img,imgpts)
    
    return img

def draw_3d_cube(img, points):
    imgpts = np.int32(points).reshape(-1,2)
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,),3)

    return img


if __name__ == "__main__":
    main()