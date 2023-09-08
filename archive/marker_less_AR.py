import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class pattern:
    size: tuple
    frame: np.array
    gray: np.array
    keypoints: tuple
    descriptors: np.array
    points2d: np.array
    points3d: np.array

#######################################################################

class markerless_ar():
    def __init__(self):
        # ORB for both dectector and extractor, Brute-Force matcher for matcher
        self.dectector = cv2.ORB().create(nfeatures=1000)
        self.extractor = self.dectector
        # self.extractor = cv2.xfeatures2d.FREAK().create(False, False)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # define the parameters for matching
        self.MIN_MATCH_COUNT = 10
        self.ransacReprojThreshold = 1

        retvel, self.kp_markers, self.des_markers = self.Descriptor_All_Markers()

        # self.marker_img = cv2.imread("scripts/markers_images/marker_2.png")
        # self.pattern_create()


    def Descriptor_All_Markers(self):
        """
        create the descriptor for the pre-defined markers. The number of markers is 1-7.
        This function will return 6 pairs of key points and descriptor of the markers
        """
        des_markers = []
        kp_markers = []
        self._ids_original = [3,6,7,12,13,14]
        for i in range(len(self._ids_original)):
            name = "scripts/markers_images/marker_" + str(self._ids_original[i]) + ".png"
            marker_img = cv2.imread(name)
            _gray_marker = cv2.cvtColor(marker_img, cv2.COLOR_BGR2GRAY)
            kp, des = self.feature_detect_extract(_gray_marker)

            # store the keypoints and descriptors
            kp_markers.append(kp)
            des_markers.append(des)

        if len(kp_markers) != 6 or len(des_markers) != 6:
            return False, None, None
        
        return True, kp_markers, des_markers
    

    def feature_matching(self, crop_image, ids):
        """
        This function will match the feature of cropped images and the stored descriptor
        """
        # for i in range(len(ids)):
        for i in range(len(ids)):
            # detect and extract the features of croped image
            _id = ids[i][0]
            q_img = crop_image[i]
            q_kp, q_des = self.feature_detect_extract(q_img)

            # extract the corresponding keypoint and descriptor of the marker
            _index_marker = self._ids_original.index(_id)
            t_kp = self.kp_markers[_index_marker]
            t_des = self.des_markers[_index_marker]
            t_img = cv2.imread("scripts/markers_images/marker_" + str(_id) + ".png")
            _gray_t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)

            _img_without_refine, homo = self.match_and_refine(q_des,t_des,q_kp,t_kp,q_img,_gray_t_img)
            # cv2.imshow("marker_"+str(_id), _img_without_refine)

            if homo is not None:
                warp_img = cv2.warpPerspective(q_img, homo, (150,150),flags=cv2.WARP_INVERSE_MAP)
                # detect and extract from the warpped image
                warp_kp, warp_des = self.feature_detect_extract(warp_img)
                _img, homo_enhanced = self.match_and_refine(warp_des,t_des,warp_kp,t_kp,warp_img,_gray_t_img)
                cv2.imshow("marker_"+str(_id), _img)

    

    def feature_detect_extract(self, src_img):
        if src_img is not None:
            kp = self.dectector.detect(src_img)
            kp, des = self.extractor.compute(src_img, kp)
            return kp, des
        else:
            print("input image is empty")
            return None, None


    def match_and_refine(self, query_des, train_des, query_kp, train_kp, query_img, train_img):
        """
        query_des and query_kp are the descriptor and keypoints of croped images.
        train_des and train_kp are the descriptor and keypoints of markers images.
        """
        des_cam = query_des
        des_marker = train_des

        kp_cam = query_kp
        kp_marker = train_kp

        cam_img = query_img
        marker_img = train_img

        _img = cam_img
        self.matcher.clear()

        if (des_cam is not None) and (des_marker is not None):
            _matches = self.matcher.match(des_cam, des_marker)
            if len(_matches) > self.MIN_MATCH_COUNT: 
                src_pts = np.float32([kp_cam[m.queryIdx].pt for m in _matches]).reshape(-1,1,2)
                dst_pts = np.float32([kp_marker[m.trainIdx].pt for m in _matches]).reshape(-1,1,2)
                
                # retval is the homography matrix
                retval, inliermask = cv2.findHomography(srcPoints=src_pts, dstPoints=dst_pts, 
                                                        method=cv2.RANSAC, 
                                                        ransacReprojThreshold=self.ransacReprojThreshold)
                
                inliers = []
                for i in range(len(inliermask)):
                    if (inliermask[i]):
                        inliers.append(_matches[i])
                inliers_matches = tuple(inliers)

                _img_match = cv2.drawMatches(img1=cam_img, keypoints1=kp_cam, 
                                       img2=marker_img, keypoints2=kp_marker, 
                                       matches1to2=inliers_matches[:1], outImg=None, 
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                print("outlier removed")
                return _img_match, retval
            else:
                print("total marches are too less")
                return _img, None
        else:
            print("no match")
            return _img, None



    def pattern_create(self):

        gray = cv2.cvtColor(self.marker_img, cv2.COLOR_BGR2GRAY)
        kp = self.dectector.detect(image=gray)
        kp, des = self.extractor.compute(image=gray, keypoints=kp)

        # width and height of the image
        w = gray.shape[1]
        h = gray.shape[0]
        unitw = w/max(w,h)
        unith = h/max(w,h)

        point2d = np.array([[0,0], [w,0], [w,h], [0,h]])
        point3d = np.array([[-unitw, -unith, 0],
                           [unitw, -unith, 0],
                           [unitw, unith, 0],
                           [-unitw, unith, 0]])

        self.pattern_marker_2 = pattern(size= self.marker_img.shape,
                                       frame= self.marker_img,
                                       gray=gray,
                                       keypoints=kp,
                                       descriptors=des,
                                       points2d=point2d,
                                       points3d=point3d
                                       )
        

    def crop_marker(self, current_frame):
        """
        current_frame is the grayscale image from the camera
        """
        gray = current_frame
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        arucoParams = cv2.aruco.DetectorParameters()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, dictionary, parameters=arucoParams)
        
        crop_img = []
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
                    
                    # print("_x", _x)
                    # print("_y", _y)
                    # print(top_left_x,top_left_y,bottom_left_x,bottom_left_y)

                    for j in range(len(top_left_x)):
                        _inter_img = gray[top_left_y[j]:bottom_right_y[j]+1, top_left_x[j]:bottom_right_x[j]+1]
                        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                        _inter_img = cv2.filter2D(_inter_img, -1, kernel=kernel)
                        alpha = 5
                        beta = 1
                        _inter_img = cv2.convertScaleAbs(_inter_img, alpha=alpha, beta=beta)
                        crop_img.append(_inter_img)

        return crop_img, ids


def main():
    AR = markerless_ar()

    # dertermind the camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret,img = cap.read()
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            crop_markers, ids = AR.crop_marker(gray)

            if ids is not None:
                for i in range(len(ids)):
                    AR.feature_matching(crop_markers, ids)
            cv2.imshow("camera_frame", img)
            cv2.waitKey(1)



if __name__ == "__main__":
    main()
    # test = markerless_ar()
    # print(type(test.pattern_marker_2.frame))
    