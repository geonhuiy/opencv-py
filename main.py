import cv2
import numpy as np
import argparse
from sizeCheck import SizeCheck

# For CLI arguments
ap = argparse.ArgumentParser()
# Filepath to image is necessary to run, colour is not implemented yet.
ap.add_argument('-i', '--image', help='Path to image')
ap.add_argument('-c', '--colour', help='Colour to detect')
args = vars(ap.parse_args())
img = cv2.imread(args["image"])


def main():
    # Opens the image from the given filepath in argument
    #filterColour()
    SizeCheck.checkLargest(img)


def filterColour():

    # Converts BGR to HSV for more accurate reading
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # A hardcoded min-max colour range for GREEN
    img_mask = cv2.inRange(img_hsv, (35, 0, 0), (75, 255, 255))

    # Filters and applies mask to show output
    output = cv2.bitwise_and(img, img, mask=img_mask)
    cv2.imshow('green', output)

    # Output window is closed if any key is pressed
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
