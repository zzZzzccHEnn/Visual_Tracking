import cv2
import numpy as np
import math


def main():
    # load the image
    img = cv2.imread("scripts/images/img0.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # print(img.shape)

    width = 9
    height = 6

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((height*width,3), np.float32)
    objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    _ret, corners = cv2.findChessboardCorners(gray, (width,height), None)

    blank_image = np.zeros((img.shape[0],img.shape[1],3), np.uint8)


    if _ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, (width,height), corners2, _ret)

        # print("objectpoint,", objpoints)
        # print("imagepoint,",imgpoints)

        _, _mtx, _dist, _rvecs, _tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # convert the _tvec to tuple
        tuple_tvecs = []
        for i in _tvecs[0]:
            tuple_tvecs.append(i[0])
        _tvecs = tuple(tuple_tvecs)

        # reproject the object points
        re_project, _ = cv2.projectPoints(np.array(objpoints), np.eye(3), _tvecs, _mtx, _dist)
        # print(re_project)

        int_re_project = np.zeros((len(re_project), 2), dtype="float64")
        # print(np.shape(int_re_project))

        # print(np.shape(re_project))
        # print(re_project[-1][0][0])

        for j in range(len(re_project)):
            # int_re_project[j][0] = int(np.round(re_project[j][0][0], decimals=0))
            # int_re_project[j][1] = int(np.round(re_project[j][0][1], decimals=0))
            int_re_project[j][0] = np.round(re_project[j][0][0])
            int_re_project[j][1] = np.round(re_project[j][0][1])
        # print(int_re_project)
        int_re_project = int_re_project.astype(np.int64)

        print(int_re_project)
        # print((int_re_project[-1][0]))

        for k in range(len(re_project)):
            blank_image = cv2.circle(blank_image, int_re_project[k], radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imshow("test", blank_image)
            cv2.waitKey(1000)


        # intrinsic matrix is _mtx
        # create the extrinsic matrix
        # extrin = np.zeros((3,4))
        # extrin[0:3,0:3] = np.eye(3)
        # extrin[:,3] = np.array(_tvecs).reshape(3,)
        # # print(_mtx @ extrin)

        # for i in range(len(objpoints)):
        #     # print(objpoints[0][i])
        #     np.append(objpoints[0][i], 1)
        # print(objpoints)




if __name__ == "__main__":
    main()