import cv2
import sys
import numpy as np
import math

print("Args: " +  str(sys.argv))
if(len(sys.argv) != 2):
    quit()
print("Hi")

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#img = cv2.GaussianBlur(img,(9,9),0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

edges = cv2.Canny(img,300,400)
#edges = cv2.Canny(img,100,200)

cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLines(edges,1,np.pi/180,200)

print(lines.shape)
print(lines)

index = 0
maxlines = 10
intervals = 180
slotwidth = 360 // intervals
ortho_dist = 90 // slotwidth
if(90 % slotwidth != 0):
    print("Error, bad number of intervals. Needs to be divisible by 90 or something. Not entirely sure, math is hard")
# stores lines as tuples (x0, y0, dx=b, dy=a) based on angle
#organized_lines = [[]] * intervals
organized_lines = [[] for i in range(intervals)]

#print(organized_lines)

def pl():
    for i in range(0, len(organized_lines)):
        print(str(len(organized_lines[i])) + " ")

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
        cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

        angle = math.atan2(a, b) * 180/np.pi
        if(angle < 0):
            angle += 360

        slot = round(angle / slotwidth) % intervals
#        print("slot: " + str(slot))
#        pl()
#        print(organized_lines[slot])
        organized_lines[slot].append((x0, y0, b, a))
#        print(organized_lines[slot])
#        print(organized_lines)

for i in range(0, len(organized_lines)):
    #print(i)
    #print(ortho_dist)
    #print(intervals)
    next = (i + ortho_dist) % intervals
    #print(type(i))
    #print(type(next))
    if(len(organized_lines[i]) >= 2 and len(organized_lines[next]) >= 2):
        print(str(i) + " (" + str(i * slotwidth) + " degrees) matches " + str(next) + " (" + str(next * slotwidth) + " degrees)")
        print("lengths are " + str(len(organized_lines[i])) + " and " + str(len(organized_lines[next])))

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

