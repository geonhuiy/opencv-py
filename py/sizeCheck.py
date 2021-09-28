import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from CropLayer import CropLayer
import os


from numpy.core.shape_base import block

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='Path to image',
                required=True, type=str)
ap.add_argument('-p', '--hedPath',
                help='Path to HED caffemodel and deploy.prototxt', required=True, type=str)
args = vars(ap.parse_args())
img = cv2.imread(args["image"])


def main():
    #SizeCheck.checkArea(SizeCheck.hed(img))
    #SizeCheck.checkArea(img)
    SizeCheck.watershed(img)



class SizeCheck:

    def checkArea(img):
        blockSize = 31
        C = 3

        blur = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        thresholdMean = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,blockSize, C)
        thresholdGaussian = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
        thresholdOtsu = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Perform connected component labeling
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholdMean , connectivity=4)

        # Create false color image and color background black
        colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        false_colors = colors[labels]

        # Label area of each polygon
        false_colors_area = false_colors.copy()
        for i, centroid in enumerate(centroids[1:], start=1):
            area = stats[i, 4]
            cv2.putText(false_colors_area, str(area), (int(centroid[0]), int(
                centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        rgb_gaussian = cv2.cvtColor(thresholdGaussian, cv2.COLOR_BGR2RGB)
        rgb_otsu = cv2.cvtColor(thresholdOtsu, cv2.COLOR_BGR2RGB)
        rgb_mean = cv2.cvtColor(thresholdMean, cv2.COLOR_BGR2RGB)
        imgs = {'Original': img, 'Adapted-Gaussian': rgb_gaussian, 'Otsu': rgb_mean, 'Colors': false_colors_area}

        for i, (k, v) in enumerate(imgs.items()):
            plt.subplot(2, 2, i+1)
            plt.title(k)
            plt.imshow(v)
            plt.xticks([]), plt.yticks([])
        plt.show()
        #cv2.imshow('Original sample', img)
        #cv2.imshow('Area', false_colors_area)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

    def hed(img):
        protoPath = os.path.sep.join([args['hedPath'], 'deploy.prototxt'])
        modelPath = os.path.sep.join(
            [args['hedPath'], 'hed_pretrained_bsds.caffemodel'])
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        cv2.dnn_registerLayer('Crop', CropLayer)
        # Loads image dimensions
        (H, W) = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H), mean=(
            104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)

        net.setInput(blob)

        hed= net.forward()
        hed= cv2.resize(hed[0,0], (W,H))
        hed= (255 * hed).astype('uint8')

        return hed
    
    def watershed(img):
        #Pyramid shifting to improve accuracy
        shifted_img = cv2.pyrMeanShiftFiltering(img, 21, 51)

        gray = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow('Threshold', thresh)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
