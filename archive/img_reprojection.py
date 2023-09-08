import cv2
import numpy as np
import glob
import re


def warp(path):
    # load the image
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # print(img.shape)

    width = 9
    height = 6

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((height*width,3), np.float32)
    objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2) * 1.5 

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    _ret, corners = cv2.findChessboardCorners(gray, (width,height), None)

    blank_image = np.zeros((img.shape[0],img.shape[1],3), np.uint8)


    if _ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, (width,height), corners2, _ret)

        cv2.imshow("chessboard", img)

        # print("objectpoint,", objpoints)
        # print("imagepoint,",imgpoints)

        _error, _mtx, _dist, _rvecs, _tvecs,stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], None, None)

        # print(_error)

####################################################################

        # convert the _tvec to tuple
        # tuple_tvecs = []
        # for i in _tvecs[0]:
        #     tuple_tvecs.append(i[0])
        # _tvecs = tuple(tuple_tvecs)

        # reproject the object points
        # camera is facing the chessboard, _rvecs should be eye matrix
        re_project, _ = cv2.projectPoints(np.array(objpoints), _rvecs[0], _tvecs[0], _mtx, _dist)
        # print(re_project)

        # int_re_project = np.zeros((len(re_project), 2), dtype="float64")

        # for j in range(len(re_project)):
        #     int_re_project[j][0] = np.round(re_project[j][0][0])
        #     int_re_project[j][1] = np.round(re_project[j][0][1])

        # int_re_project = int_re_project.astype(np.int64)

        # for k in range(len(re_project)):
        #     blank_image = cv2.circle(blank_image, int_re_project[k], radius=2, color=(0, 0, 255), thickness=-1)

        img_points = np.array(imgpoints).reshape(len(imgpoints[0]),2)
        re_project = np.array(re_project).reshape(np.shape(re_project)[0],2)
        # print(type(re_project))

        # calculate the homography
        homo, mask = cv2.findHomography(srcPoints=np.array(re_project), dstPoints=img_points, method=cv2.RANSAC, ransacReprojThreshold=2)

        img_warp = cv2.warpPerspective(img, homo, (640*2,480*2),flags=cv2.WARP_INVERSE_MAP)

        # cv2.imshow("warp", img_warp)
        # k = cv2.waitKey(5)
        # if k == 27:
        #     cv2.destroyAllWindows()
        
        # save the warped images
        filename = re.sub("\.png$", "", path)
        filename = filename + "_warp" + ".png"
        cv2.imwrite(filename, img_warp)
        
    else:
        print(path, "no chessboard corner detected")


def main():
    # load the images
    imgs = glob.glob("scripts/images/*.png")

    for fname in imgs:
        if fname.endswith("warp.png"):
            continue
        else:
            warp(fname)



if __name__ == "__main__":
    main()