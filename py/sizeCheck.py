import cv2
import numpy as np
import matplotlib.pyplot as plt

from numpy.core.shape_base import block


class SizeCheck:
    # def __init__(self, img):
    #    self.img = img

    def checkLargest(img):

        blockSize =51
        C = 5

        blur = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        #thresholdMean = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,blockSize, C)
        thresholdGaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
        final = cv2.cvtColor(thresholdGaussian, cv2.COLOR_BGR2RGB)
        #cv2.imshow('Thresh-Mean', thresholdMean)

        # Perform connected component labeling
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholdGaussian, connectivity=4)

        # Create false color image and color background black
        colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        false_colors = colors[labels]

        # Label area of each polygon
        false_colors_area = false_colors.copy()
        for i, centroid in enumerate(centroids[1:], start=1):
            area = stats[i, 4]
            cv2.putText(false_colors_area, str(area), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

        imgs = {'Original': img, 'Adapted-Gaussian': false_colors_area}
        #for i, (k, v) in enumerate(imgs.items()):
        #    plt.subplot(2, 2, i+1)
        #    plt.title(k)
        #    plt.imshow(v)
        #    plt.xticks([]), plt.yticks([])
        #plt.show()
        cv2.imshow('Original sample', img)
        cv2.imshow('Area',false_colors_area)
        cv2.waitKey()
        cv2.destroyAllWindows()
