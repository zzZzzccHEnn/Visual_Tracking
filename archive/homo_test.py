import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    cap = cv2.VideoCapture(2)
    arucoParams = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    _dist = np.load("scripts/distortion_coefficients.npy")
    _mtx = np.load("scripts/camera_matrix.npy")

    

    while cap.isOpened():
        ret, img_1 = cap.read()
        if ret:
            # img_1 = cv2.imread("scripts/3.jpg")
            h, w = img_1.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(_mtx, _dist, (w,h), 1, (w,h))
            dst = cv2.undistort(img_1, _mtx, _dist, None, newcameramtx)
            x, y, w, h = roi
            img_1_undist = dst[y:y+h, x:x+w]

            img_2 = cv2.imread("scripts/markers_images/marker_13.png")
    
            img_1_gray = cv2.cvtColor(img_1_undist, cv2.COLOR_BGR2GRAY)
            img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            cv2.imshow("marker", img_2_gray)

            corners, ids, rejected = cv2.aruco.detectMarkers(img_1_gray, dictionary, parameters=arucoParams)

            test_img = img_1_gray

            if len(corners) > 0:
                for i in range(len(ids)):
                    _x = []
                    _y = []
                    for i in range(np.shape(corners)[0]):
                        _x.append([x for [x, y] in corners[i][0]])
                        _y.append([y for [x, y] in corners[i][0]])

                    top_left_x = np.array([min(x) for x in _x]).astype(int)
                    top_left_y = np.array([min(y) for y in _y]).astype(int)
                    bottom_right_x = np.array([max(x) for x in _x]).astype(int)
                    bottom_right_y = np.array([max(y) for y in _y]).astype(int)

                    warp_img = []
                    for j in range(len(top_left_x)):
                        warp_img.append(img_1_gray[top_left_y[j]:bottom_right_y[j]+1, top_left_x[j]:bottom_right_x[j]+1])

    # print(img_1_gray.shape)
    # cv2.imshow("test", img_1_gray)
    # cv2.waitKey(2000)
    # img_1 = img_1.astype(cv2.cv)

                    test_img = warp_img[0]
                    cv2.imshow("crop", test_img)
            
            orb = cv2.ORB().create(nfeatures=1000)
            # extractor = cv2.xfeatures2d.FREAK().create()
            extractor = orb

    # kp_1 = orb.detect(img_1_gray)
            # kp_1 = orb.detect(img_1_gray)
            # kp_1, des_1 = extractor.compute(img_1_gray, kp_1)
            kp_1 = orb.detect(test_img)
            kp_1, des_1 = extractor.compute(test_img, kp_1)
            kp_2 = orb.detect(img_2_gray)
            kp_2, des_2 = extractor.compute(img_2_gray, kp_2)

    # matcher
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matcher.clear()

            if (des_1 is not None) and (des_2 is not None): 
                _matches = matcher.match(des_1, des_2)
                # good = []
                # # print(np.ndim(_matches))
                # for m,n in _matches:
                #     if m.distance < 0.66*n.distance:
                #         good.append([m])

                # print(len(good))
                # print("number of matches",len(_matches))
                if len(_matches) > 8: 
                    src_pts = np.float32([kp_1[m.queryIdx].pt for m in _matches]).reshape(-1,1,2)
                    dst_pts = np.float32([kp_2[m.trainIdx].pt for m in _matches]).reshape(-1,1,2)

                    homo, inliermask = cv2.findHomography(srcPoints=src_pts, dstPoints=dst_pts, 
                                                    method=cv2.RANSAC, 
                                                    ransacReprojThreshold=2)

                    inliers = []
                    for i in range(len(inliermask)):
                        if (inliermask[i]):
                            inliers.append(_matches[i])
                    inliers_matches = tuple(inliers)

                    _img_warp = cv2.warpPerspective(img_2_gray, homo,(640,480),flags=cv2.WARP_INVERSE_MAP)

                    # solvePnP

                    # _, rvec, tvec = cv2.solvePnP()

                    _img_match = cv2.drawMatches(img1=test_img, keypoints1=kp_1, 
                                       img2=img_2, keypoints2=kp_2, 
                                       matches1to2=inliers_matches[:1], outImg=None, 
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # plt.imshow(_img)
        # plt.show()
                    cv2.imshow("warp", _img_warp)
                    cv2.imshow("match", _img_match)

        cv2.imshow("camera", img_1)
        cv2.waitKey(5)


if __name__ == "__main__":
    main()