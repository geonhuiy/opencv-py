import cv2
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='Path to image')
ap.add_argument('-c', '--colour', help='Colour to detect')
args = vars(ap.parse_args())


def main():

    img = cv2.imread(args["image"])
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, (35, 0, 0), (75, 255, 255))
    output = cv2.bitwise_and(img, img, mask=img_mask)
    cv2.imshow('green', output)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
