import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
from numpy.core.numeric import count_nonzero
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from numpy.core.shape_base import block

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='Path to image',
                required=True, type=str)
# ap.add_argument('-p', '--hedPath',
#                help='Path to HED caffemodel and deploy.prototxt', required=True, type=str)
args = vars(ap.parse_args())
img = cv2.imread(args["image"])


def main():
    connectedComp(img)


def displayImage(img, title):
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def connectedComp(img):
    blockSize = 31
    C = 3

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresholdMean = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)
    thresholdGaussian = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C)
    #displayImage(threshold, 'Threshold')
    #displayImage(thresholdMean, 'Mean')
    #drawContour(thresholdGaussian)
    contours = cv2.findContours(thresholdGaussian, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    largest = max(contours, key=cv2.contourArea)
    img_edge = cv2.drawContours(img, [largest], -1, (0,255,0),2)
    displayImage(img, 'Gaussian')

def drawContour(img):

    contours = cv2.findContours(img.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = contours[0] if len(contours) == 2 else contours[1]
    largestContour = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [largestContour], 1 ,(0,255,0),2)

if __name__ == '__main__':
    main()
