import numpy as np
import cv2

img = cv2.imread('RAS_RGB_img/1.jpg')

himg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

height, width = himg.shape[:2]
himg = cv2.resize(himg, (int(0.3*width), int(0.3*height)), interpolation=cv2.INTER_CUBIC)
img = cv2.resize(img, (int(0.3*width), int(0.3*height)), interpolation=cv2.INTER_CUBIC)
Z = himg.reshape((-1, 3))
Y = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)
Y = np.float32(Y)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret1, label1, center1 = cv2.kmeans(Z, 12, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
ret2, label2, center2 = cv2.kmeans(Y, 10, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center1 = np.uint8(center1)
res = center1[label1.flatten()]
res1 = res.reshape(himg.shape)

center2 = np.uint8(center2)
res = center2[label2.flatten()]
res2 = res.reshape(img.shape)

cv2.imshow('HSV-kmeans', res1)
cv2.imshow('RGB-kmeans', res2)
cv2.imshow('original', img)
cv2.imshow('HSV-original', himg)
#cv2.imshow('grey-HSV-kmeans', gimg)

"""get an idea about what the following lines are from 
https://www.learnopencv.com/blob-detection-using-opencv-python-c/"""

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 1;
params.maxThreshold = 255;

# Filter by Area.
params.filterByArea = True
params.minArea = 500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.05

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.3

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.3

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

gimg = cv2.cvtColor(cv2.cvtColor(res1, cv2.COLOR_HSV2BGR_FULL), cv2.COLOR_BGR2GRAY)

# Detect blobs.
keypoints = detector.detect(gimg)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(gimg, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)

"""above blob detection only implement grey image blob detection, you can find roughly 
how to do that from http://www.shervinemami.info/blobs.html"""

cv2.waitKey(0)
cv2.destroyAllWindows()

