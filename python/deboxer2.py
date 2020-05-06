from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage.segmentation as seg
import cv2
import skimage.draw as draw
import skimage.color as color

print("Args: " +  str(sys.argv))
if(len(sys.argv) != 2):
    quit()

cap = cv2.VideoCapture(sys.argv[1])

counter = -1
speedinator = 30

while(cap.isOpened()):

    #image = cv2.imread(sys.argv[1])

    ret, image = cap.read()

    counter += 1
    if(counter % speedinator != 0):
        continue

    image = cv2.resize(image, (1920, 1080))

    image_slic = seg.slic(image,n_segments=155)
    #image_slic = seg.slic(image,n_segments=50)
    image_slic = image_slic.astype(np.uint8)

    colored = color.label2rgb(image_slic, image, kind='avg')
    colored = colored.astype(np.uint8)
    cv2.imshow('img', colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    import math
    from matplotlib import pyplot as plt 

    img = image_slic


    edges = cv2.Canny(img,5,100)

    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lines = cv2.HoughLines(edges,1,np.pi/180,150)

    index = 0
    maxlines = 1000
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
            cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)


            angle = math.atan2(a, b) * 180/np.pi
            if(angle < 0):
                angle += 360

            slot = round(angle / slotwidth) % intervals
            organized_lines[slot].append((x0, y0, b, a))

    #print(organized_lines)

    for i in range(0, len(organized_lines)):
        next = (i + ortho_dist) % intervals
        if(len(organized_lines[i]) >= 2 and len(organized_lines[next]) >= 2):
            print(str(i) + " (" + str(i * slotwidth) + " degrees) matches " + str(next) + " (" + str(next * slotwidth) + " degrees)")
            print("lengths are " + str(len(organized_lines[i])) + " and " + str(len(organized_lines[next])))

    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

