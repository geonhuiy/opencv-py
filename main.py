import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "Path to image")
args = vars(ap.parse_args())
img = cv2.imread(args["image"])
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
