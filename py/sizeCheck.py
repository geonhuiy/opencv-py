import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
from numpy.core.numeric import count_nonzero
from skimage.feature import peak, peak_local_max
from skimage.segmentation import watershed
from skimage import color, io
from scipy import ndimage
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
    # SizeCheck.checkArea(img)
    # SizeCheck.watershed(img)
    # SizeCheck.hed(img)
    # SizeCheck.segment_hed(img)
    # SizeCheck.auto_canny(img)
    SizeCheck.watershed_new(img)


class SizeCheck:

    def checkArea(img):
        blockSize = 31
        C = 3

        blur = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        thresholdMean = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)
        thresholdGaussian = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
        thresholdOtsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Perform connected component labeling
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresholdMean, connectivity=4)

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
        imgs = {'Original': img, 'Adapted-Gaussian': rgb_gaussian,
                'Mean': rgb_mean, 'Colors': labels}

        for i, (k, v) in enumerate(imgs.items()):
            plt.subplot(2, 2, i+1)
            plt.title(k)
            plt.imshow(v)
            plt.xticks([]), plt.yticks([])
        plt.show()
        #cv2.imshow('Original sample', img)
        #cv2.imshow('Area', false_colors_area)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    # HED, deep learning edge detection
    def hed(img):
        protoPath = os.path.sep.join([args['hedPath'], 'deploy.prototxt'])
        modelPath = os.path.sep.join(
            [args['hedPath'], 'hed_pretrained_bsds.caffemodel'])
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        cv2.dnn_registerLayer('Crop', CropLayer)
        # Loads image dimensions

        (H, W) = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H), mean=(
            104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)

        net.setInput(blob)

        hed = net.forward()
        resize_scale = 0.4
        width = int(W * resize_scale)
        height = int(H * resize_scale)
        hed = cv2.resize(hed[0, 0], (width, height))
        hed = (255 * hed).astype('uint8')
        contours = cv2.findContours(
            hed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        print(len(contours))
        cv2.imshow('HED', hed)
        cv2.imshow('Original sample', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        # return hed

    # Watershed for object segmentation
    def watershed(img):
        #blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
        # Pyramid shifting to improve accuracy
        shifted_img = cv2.pyrMeanShiftFiltering(img, 1, 1)
        gray = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Adjust min_distance for contour area sensitivity
        D = ndimage.distance_transform_edt(thresh)
        localmax = peak_local_max(
            D, indices=False, min_distance=14, labels=thresh)

        markers = ndimage.label(localmax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)
        print(len(labels))
        for label in np.unique(labels):
            # Lable 0 is background, ignored
            if label == 0:
                continue

            # Allocate memory for label region and draw on mask
            mask = np.zeros(gray.shape, dtype='uint8')
            mask[labels == label] = 255

            # Detect largest contour in the mask
            contours = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours = imutils.grab_contours(contours)z
            contours = contours[0] if len(contours) == 2 else contours[1]
            largestContour = max(contours, key=cv2.contourArea)

            # cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
            #    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
            cv2.drawContours(img, [largestContour], -1, (36, 255, 12), 2)
        cv2.imshow('Area', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def watershed_new(img):
        blur = cv2.GaussianBlur(img, (3,3), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        background = cv2.dilate(opening, kernel, iterations=3)
        distance_transform = cv2.distanceTransform(opening, cv2.DIST_L2,3)
        ret2, foreground = cv2.threshold(distance_transform, 0.21*distance_transform.max(), 255,0)
        #D = ndimage.distance_transform_edt(thresh)
        #localmax = peak_local_max(D, indices=False, min_distance=15, labels=thresh)
        #markers = ndimage.label(localmax, structure=np.ones((3,3)))[0]
        #labels = watershed(-D, markers, mask= thresh)

        #for label in np.unique(labels):
        #    if label == 0:
        #        continue
        #    mask = np.zeros(gray.shape, dtype='uint8')
        #    mask[labels == label] = 255
        #    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #    cnts = imutils.grab_contours(cnts)
        #    C = max(cnts, key=cv2.contourArea)
        #    print(C[0])
        #    cv2.drawContours(img, [C], -1, (0,255,0),2)
        foreground = np.uint8(foreground)
        unknown = cv2.subtract(background, foreground)
        marker_count, markers = cv2.connectedComponents(foreground)
        # Separates foreground values from background values
        markers = markers+1
        markers[unknown==255] = 0
        ws_segment = cv2.watershed(img, markers)
        img[markers == -1 ] = [0,255,0]
        img_output = color.label2rgb(markers, bg_label=1)
        output_ws = np.zeros_like(img)
        output2 = img.copy()

        area_list = []
        largest_area = 0
        for i in range(2,marker_count + 1):
            mask = np.where(ws_segment==i, np.uint8(255),np.uint8(0))
            x,y,w,h = cv2.boundingRect(mask)
            area = cv2.countNonZero(mask[y:y+h,x:x+w])
            area_list.append(area)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #C = max(cnts, key=cv2.contourArea)
            #cv2.drawContours(img,C,-1(0,255,0),2)
        
        print('Largest area : ', max(area_list), 'Smallest area : ', (min(i for i in area_list if i > 0)))


        #labels = watershed(-distance_transform, markers, mask=thresh)
        #contours = list(map(lambda l: cv2.findContours((labels == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], np.unique(labels)[1:]))
        #biggest_contour = max(contours, key=cv2.contourArea)
        #smallest_contour = min(contours, key=cv2.contourArea)
        #print(biggest_contour, smallest_contour)
        #cv2.drawContours(img_biggest, contours, -1,(0,255,0),2)
        cv2.imshow('Original sample', img)
        cv2.imshow('Watershed segmented', img_output)
        cv2.waitKey()
        cv2.destroyAllWindows()
        

    # Canny used for edge detection
    def auto_canny(img, sigma=0.33):
        img_biggest = img.copy()
        img_smallest = img.copy()
        # Median of single channel pixel of image
        v = np.median(img)

        # Automatic canny thresholds based on the median of the image
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        # Gray and blur to remove noise from image
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (7, 7), 0)
        #bilateral = cv2.bilateralFilter(img, 2, 20 ,20 )
        edge = cv2.Canny(blur, lower, upper)
        contours = cv2.findContours(edge.copy(),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        biggest_contour = max(contours, key=cv2.contourArea)
        smallest_contour = min(contours, key=cv2.contourArea)
        #cv2.drawContours(img, biggest_contour, -1, (0, 255, 0), 2)
        cv2.drawContours(img_biggest, [biggest_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(img_smallest, [smallest_contour], -1, (0, 255, 0), 2)

        cv2.imshow('Original sample image', img)
        cv2.imshow('Biggest pebble', img_biggest)
        cv2.imshow('Smallest pebble', img_smallest)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def find_largest_area(array):
        largest = 0
        for x in range(0, len(array)):
            if(array[x] > largest):
                largest: array[x]
        return largest


if __name__ == '__main__':
    main()
