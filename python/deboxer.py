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

#edges = cv2.Canny(img,300,400)
edges = cv2.Canny(img,100,200)

cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLines(edges,1,np.pi/180,200)

print(lines.shape)
print(lines)

index = 0
maxlines = 1000000

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


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
