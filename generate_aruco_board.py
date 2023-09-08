import cv2
import numpy as np
import os

# cv2.__version__ = 4.8

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.GridBoard(size=(6,9), markerLength=0.01, markerSeparation=0.001, 
                            dictionary=dictionary, ids=np.arange(0,54))
img = board.generateImage((2160,3840))
name = "scripts/aruco_board_demo"+".png"
cv2.imwrite(name, img)



