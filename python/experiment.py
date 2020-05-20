from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage.segmentation as seg
import cv2
import skimage.draw as draw
import skimage.color as color
from sklearn.cluster import MeanShift, estimate_bandwidth
from moviepy.editor import *
import math

print("Args: " +  str(sys.argv))
if(len(sys.argv) != 2):
    quit()

cap = cv2.VideoCapture(sys.argv[1])
numframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

counter = -1
speedinator = 10

thresh = 10
val = -30

allimages = []
first = True
total_image = 0

while(cap.isOpened()):

    counter += 1

    #image = cv2.imread(sys.argv[1])

    ret, image = cap.read()


    print("Frame " + str(counter) + "/" + str(numframes))

    if(counter % speedinator != 0):
        continue

    if(image is None):
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #sobelx8u = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=15)
    # Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
    #sobelx64f = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    #abs_sobelx64f = np.absolute(sobelx64f)
    #sobelx_8u = np.uint8(abs_sobelx64f)

    #sobely64f = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    #abs_sobely64f = np.absolute(sobely64f)
    #sobely_8u = np.uint8(abs_sobely64f)

    laplacian = cv2.Laplacian(image, cv2.CV_8U)
    laplacian = laplacian.astype(int)
    laplacian[laplacian < thresh] = val

    if(first):
        total_image = laplacian
        first = False
    else:
        total_image += laplacian


    #ret, laplacian = cv2.threshold(laplacian,127,255,cv2.THRESH_BINARY)

    allimages.append(laplacian)

    #cv2.imshow('img', sobel_8u)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#print(total_image)

#av = np.mean(allimages, axis=0)
total_image = total_image.astype(float)
total_image /= counter

#print(total_image)

#print(allimages[0].shape)
#print(av.shape)
total_image = np.clip(total_image, 0, 255)

total_image *= (255 / total_image.max())

total_image = total_image.astype(np.uint8)

#print(total_image)

cv2.imshow('img', total_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

total_image = cv2.threshold(total_image, 20, 255, cv2.THRESH_BINARY)[1]
#edges = cv2.Canny(total_image,0,0)
edges = total_image

cv2.imshow('img', total_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

width, height = total_image.shape[:2]
linesize = int(min(width, height)/8)

padding = int(min(width, height)/8)


def maxsubarray(arr):
    #print("SHAPE" + str(arr.shape))

    arr = arr.ravel()

    s = arr.shape[0]

    #print(arr.shape)
    #print(s)

    dp = [0 for i in range(s)]
    dp = np.asarray(dp, dtype=int)

    prev = [0 for i in range(s)]
    prev = np.asarray(prev, dtype=int)

    dp[0] = arr[0]
    for i in range(1, s):
        #print("REMOVE")

        cont = dp[i-1]+arr[i]
        new = arr[i]

        if(cont > new):
            dp[i] = cont
            prev[i] = prev[i - 1]
        else:
            dp[i] = new
            prev[i] = i

        #dp[i] = max(dp[i-1]+nums[i],nums[i])

    end = dp.argmax(axis = 0)
    start = prev[end]

    #print(arr)
    #print("==========================================================================================================")
    #print(dp)
    #print("==========================================================================================================")
    #print(prev)
        #return max(dp)
    return (start, end)

#flat = np.sum(edges, axis = 0)
#flat[0] = 0
#s = flat.shape[0]
#flat[s - 1] = 0
#print("Maxcol: " + str(flat.argmax(axis=0)) + "/" + str(s))
#print(flat)
#print("==========================================================================================================")

edges = edges.astype(int)
edges[edges == 0] = -255

for i in range(width):
    col = edges[:, i]
    start, end = maxsubarray(col)
    if(end - start > 20):
        print("Col " + str(i) + "/" + str(width) + ": " + str(start) + " -> " + str(end))

#lines = cv2.HoughLines(edges,1,np.pi/180,linesize)
#
#tolerance = 1
#index = 0
#maxlines = 220
#intervals = 180
#slotwidth = 360 // intervals
#ortho_dist = 90 // slotwidth
#if(90 % slotwidth != 0):
#    print("Error, bad number of intervals. Needs to be divisible by 90 or something. Not entirely sure, math is hard")
## stores lines as tuples (x0, y0, dx=b, dy=a) based on angle
#
#xs = []
#ys = []
#
#print(lines.shape)
#
#if lines is not None:
#    for i in range(0, len(lines)):
#        index += 1
#        if(index > maxlines):
#            break
#
#        rho = lines[i][0][0]
#        theta = lines[i][0][1]
#        a = math.cos(theta)
#        b = math.sin(theta)
#        x0 = a * rho
#        y0 = b * rho
#        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#        #cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
#
#        angle = math.atan2(a, b) * 180/np.pi
#        if(angle < 0):
#            angle += 360
#
#
#
#        horiz = (angle <= tolerance or angle >= 360 - tolerance or abs(angle - 180) <= tolerance)
#        vert  = (abs(angle - 90) <= tolerance or abs(angle - 270) <= tolerance)
#
#        valid = (horiz and y0 > padding and y0 < height - padding) or (vert and x0 > padding and x0 < width - padding)
#
#        if(valid and (horiz or vert)):
#            cv2.line(total_image, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)
#
#        if(valid and horiz):
#            ys.append(y0)
#        if(valid and vert):
#            xs.append(x0)
#
#cv2.imshow('img', total_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
