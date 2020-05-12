from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage.segmentation as seg
import cv2
import skimage.draw as draw
import skimage.color as color
from sklearn.cluster import MeanShift, estimate_bandwidth

print("Args: " +  str(sys.argv))
if(len(sys.argv) != 2):
    quit()

cap = cv2.VideoCapture(sys.argv[1])
numframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

counter = -1
speedinator = 10

debug = False

ys = []
xs = []

bestx = -1
besty = -1

while(cap.isOpened()):

    #image = cv2.imread(sys.argv[1])

    ret, image = cap.read()

    print("Frame " + str(counter) + "/" + str(numframes))

    counter += 1
    if(counter % speedinator != 0):
        continue

    if(image is None):
        break

    #image = cv2.resize(image, (1920, 1080))
    # FIXME swap
    width, height = image.shape[:2]
    linesize = int(min(width, height)/4)


    image_slic = seg.slic(image,n_segments=155)
    #image_slic = seg.slic(image,n_segments=30)
    image_slic = image_slic.astype(np.uint8)

    colored = color.label2rgb(image_slic, image, kind='avg')
    colored = colored.astype(np.uint8)
    
    if(debug):
        cv2.imshow('img', colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    import math
    from matplotlib import pyplot as plt 

    img = image_slic


    edges = cv2.Canny(img,0,0)

    if(debug):
        cv2.imshow('edges', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    lines = cv2.HoughLines(edges,1,np.pi/180,linesize)

    tolerance = 1
    index = 0
    maxlines = 220
    intervals = 180
    slotwidth = 360 // intervals
    ortho_dist = 90 // slotwidth
    if(90 % slotwidth != 0):
        print("Error, bad number of intervals. Needs to be divisible by 90 or something. Not entirely sure, math is hard")
    # stores lines as tuples (x0, y0, dx=b, dy=a) based on angle
    organized_lines = [[] for i in range(intervals)]

    if lines is not None:
        for i in range(0, len(lines)):
            index += 1
            if(index > maxlines):
                break

            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            #cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

            angle = math.atan2(a, b) * 180/np.pi
            if(angle < 0):
                angle += 360

            

            horiz = (angle <= tolerance or angle >= 360 - tolerance or abs(angle - 180) <= tolerance)
            vert  = (abs(angle - 90) <= tolerance or abs(angle - 270) <= tolerance)  

            if(horiz or vert):
                cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

            if(horiz):
                ys.append(y0)
            if(vert):
                xs.append(x0)


            slot = round(angle / slotwidth) % intervals
            organized_lines[slot].append((x0, y0, b, a))

    #print(organized_lines)

#    for i in range(0, len(organized_lines)):
#        next = (i + ortho_dist) % intervals
#        if(len(organized_lines[i]) >= 2 and len(organized_lines[next]) >= 2):
#            print(str(i) + " (" + str(i * slotwidth) + " degrees) matches " + str(next) + " (" + str(next * slotwidth) + " degrees)")
#            print("lengths are " + str(len(organized_lines[i])) + " and " + str(len(organized_lines[next])))

    if(debug):
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("Sizes")
print(len(xs))
print(len(ys))

cap = cv2.VideoCapture(sys.argv[1])
ret, image = cap.read()

# FIXME swap
width, height = image.shape[:2]

xs = np.asarray(xs, dtype=np.float32)
ys = np.asarray(ys, dtype=np.float32)

cx = MeanShift(bandwidth = width/100).fit(xs.reshape(-1, 1))
cy = MeanShift(bandwidth = height/100).fit(ys.reshape(-1, 1))


print(cx.cluster_centers_)
print(cy.cluster_centers_)

print("Counts")
print(np.bincount(cx.labels_))
print(np.bincount(cy.labels_))

mx_ind = np.argmax(np.bincount(cx.labels_), axis=0)
my_ind = np.argmax(np.bincount(cy.labels_), axis=0)

mx = cx.cluster_centers_[mx_ind]
my = cy.cluster_centers_[my_ind]

print("xs, max ind, count is ")
print(mx_ind)
print(np.bincount(cx.labels_)[mx_ind])
for i in range(0, cx.cluster_centers_.shape[0]):
    print("cur is " + str(np.bincount(cx.labels_)[i]))

    if(np.bincount(cx.labels_)[i] < np.bincount(cx.labels_)[mx_ind] / 2):
        continue

    print("not skipped")

    x = cx.cluster_centers_[i]

    pt1 = (x, 0)
    pt2 = (x, height - 1)
    cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

for i in range(0, cy.cluster_centers_.shape[0]):

    if(np.bincount(cy.labels_)[i] < np.bincount(cy.labels_)[my_ind] / 2):
        continue

    y = cy.cluster_centers_[i]


    pt1 = (0, y)
    pt2 = (width - 1, y)
    cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

pt1 = (mx, 0)
pt2 = (mx, height - 1)
cv2.line(image, pt1, pt2, (255,0,0), 3, cv2.LINE_AA)
bestx = int(mx[0])

pt1 = (0, my)
pt2 = (width - 1, my)
cv2.line(image, pt1, pt2, (255,0,0), 3, cv2.LINE_AA)
besty = int(my[0])


cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("================================================================ END LINE DETECTION ==============================================================================")

diffcounts = [0, 0, 0, 0]
counter = -1

while(cap.isOpened()):

    #image = cv2.imread(sys.argv[1])

    ret, image = cap.read()

    print("Frame " + str(counter) + "/" + str(numframes))

    counter += 1
    if(counter % speedinator != 0):
        continue

    if(image is None):
        break

    #image = cv2.resize(image, (1920, 1080))
    # FIXME swap
    width, height = image.shape[:2]

    subimgs = [image[0:bestx, 0:besty], image[bestx:width, 0:besty], image[0:bestx, besty:height], image[bestx:width, besty:height]]

    diffs = [0, 0, 0, 0]

    # histogram stuff
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]

    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    channels = [0, 1]

    def ashist(img):
        csv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([csv], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    hists = list(map(ashist, subimgs))

    for i in range(4):
        for j in range(4):
            if(i == j):
                continue
            diffs[i] += 1 - cv2.compareHist(hists[i], hists[j], 0) # correlation

    diffcounts[diffs.index(max(diffs))] += 1

print(diffcounts)

outbox = diffcounts.index(max(diffcounts))

cap = cv2.VideoCapture(sys.argv[1])
ret, image = cap.read()
#width, height = image.shape[:2]
height, width = image.shape[:2]

xl, yl, xh, yh = (0, 0, 0, 0)

if(outbox == 0):
    xl, yl, xh, yh = (0, 0, bestx, besty)

if(outbox == 1):
    xl, yl, xh, yh = (bestx, 0, width-1, besty)
    
if(outbox == 2):
    xl, yl, xh, yh = (0, besty, bestx, height-1)
    
if(outbox == 3):
    xl, yl, xh, yh = (bestx, besty, width-1, height-1)
    

print(width)
print(height)
print(xh)
print(yh)
cv2.rectangle(image, (xl, yl), (xh, yh), (255,0,0), -1)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
