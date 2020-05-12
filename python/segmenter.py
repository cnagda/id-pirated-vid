from skimage import data
#import scikit-image
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

image = cv2.imread(sys.argv[1])

#image_felzenszwalb = seg.felzenszwalb(image)
#image_felzenszwalb = image_felzenszwalb.astype(np.uint8)
image_slic = seg.slic(image,n_segments=155)
#image_slic = seg.slic(image,n_segments=50)
image_slic = image_slic.astype(np.uint8)
#image_show(image_felzenszwalb);
#cv2.imshow('img', image_felzenszwalb)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imshow('img', image_slic)
cv2.waitKey(0)
cv2.destroyAllWindows()

colored = color.label2rgb(image_slic, image, kind='avg')
colored = colored.astype(np.uint8)
cv2.imshow('img', colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("../blobbified.png", colored)
