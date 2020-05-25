from skimage import data
import numpy as np
import sys
import cv2
from moviepy.editor import *

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

debug = False

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

if(debug):
    cv2.imshow('img', total_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

total_image = cv2.threshold(total_image, 20, 255, cv2.THRESH_BINARY)[1]
#edges = cv2.Canny(total_image,0,0)
edges = total_image

if(debug):
    cv2.imshow('img', total_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

height, width = total_image.shape[:2]
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

flat = np.sum(edges, axis = 0)
flat[0] = 0
s = flat.shape[0]
flat[s - 1] = 0
print("Flat shape: " + str(s))
#print(flat)

#plt.plot(range(s), flat)
#plt.show()

print("Maxcol: " + str(flat.argmax()) + "/" + str(s))
#print(flat)
#print("==========================================================================================================")

edges = edges.astype(int)
edges[edges == 0] = -255

maxc = -1
maxcs = -1
maxce = -1

maxr = -1
maxrs = -1
maxre = 0-1

for i in range(3, width - 3):
    col = edges[:, i]
    start, end = maxsubarray(col)

    if(end - start > maxce - maxcs):
        maxce = end
        maxcs = start
        maxc = i

    #print("Col " + str(i) + "/" + str(width) + ": " + str(start) + " -> " + str(end))

for i in range(3, height - 3):
    row = edges[i, :]
    start, end = maxsubarray(row)
    
    if(end - start > maxce - maxcs):
        maxre = end
        maxrs = start
        maxr = i


if(maxce - maxcs < linesize or maxre - maxrs < linesize):
    print("Did not detect two lines")
    quit()

cap = cv2.VideoCapture(sys.argv[1])
ret, image = cap.read()


cv2.line(image, (maxc, maxcs), (maxc, maxce), (0,0,255), 3, cv2.LINE_AA)
cv2.line(image, (maxrs, maxr), (maxre, maxr), (0,0,255), 3, cv2.LINE_AA)

if(debug):
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

wig = 2 # wiggle room for lines to overlap

right = False
left = False
up = False
down = False

if(maxrs >= maxc - wig):
    right = True

if(maxre <= maxc + wig):
    left = True

if(maxce <= maxr + wig):
    up = True

if(maxcs >= maxr - wig):
    down = True

p1 = (maxc, maxr)

p2 = -1

if(up and right):
    p2 = (width-1, 0)

if(up and left):
    p2 = (0, 0)

if(down and right):
    p2 = (width-1, height-1)

if(down and left):
    p2 = (0, height-1)

if(p2 == -1):
    print("No obvious box")
    quit()

xl = min(p1[0], p2[0])
xh = max(p1[0], p2[0])
yl = min(p1[1], p2[1])
yh = max(p1[1], p2[1])

video = VideoFileClip(sys.argv[1])
boxvideo = video.crop(x1 = xl, y1 = yl, x2 = xh, y2 = yh)

boxvideo.write_videofile("boxvideo.mp4")

def drawrect(frame):
    cv2.rectangle(frame, (xl, yl), (xh, yh), (0,0,0), -1)
    return frame

outervideo = video.fl_image(drawrect)

outervideo.write_videofile("outervideo.mp4")
