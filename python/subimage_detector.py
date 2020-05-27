import argparse
import numpy as np
import sys
import cv2
import os
from moviepy.video.io.VideoFileClip import *
from moviepy.video.fx.crop import crop


def maxsubarray(arr):
    arr = arr.ravel()

    s = arr.shape[0]

    dp = [0 for i in range(s)]
    dp = np.asarray(dp, dtype=int)

    prev = [0 for i in range(s)]
    prev = np.asarray(prev, dtype=int)

    dp[0] = arr[0]
    for i in range(1, s):
        
        cont = dp[i-1]+arr[i]
        new = arr[i]

        if(cont > new):
            dp[i] = cont
            prev[i] = prev[i - 1]
        else:
            dp[i] = new
            prev[i] = i

    end = dp.argmax(axis = 0)
    start = prev[end]

    return (start, end)

def main():
    python_folder = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.normpath(os.path.join(python_folder, ".."))
    results_dir = os.path.join(root_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    outer_path = os.path.join(results_dir, "outervideo.mp4")
    box_path = os.path.join(results_dir, "boxvideo.mp4")

    # deleting files if they already exist because query will check if exists
    # before running on both videos
    if os.path.exists(outer_path):
        os.remove(outer_path)
    if os.path.exists(box_path):
        os.remove(box_path)



    # print("Args: " +  str(sys.argv))
    if(len(sys.argv) != 2):
        quit()

    parser = argparse.ArgumentParser(
        description='Find picture in picture and save separate files'
    )

    parser.add_argument(
        'srcpath',
        type=str,
        help='path to video with potential picture in picture'
    )

    args = parser.parse_args()

    if not os.path.isfile(args.srcpath):
        print("Invalid path, exiting")
        return

    cap = cv2.VideoCapture(args.srcpath)
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

        ret, image = cap.read()


        # print("Frame " + str(counter) + "/" + str(numframes))

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

        allimages.append(laplacian)

    total_image = total_image.astype(float)
    total_image /= counter

    total_image = np.clip(total_image, 0, 255)

    total_image *= (255 / total_image.max())

    total_image = total_image.astype(np.uint8)

    if(debug):
        cv2.imshow('img', total_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    total_image = cv2.threshold(total_image, 20, 255, cv2.THRESH_BINARY)[1]
    edges = total_image

    if(debug):
        cv2.imshow('img', total_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    height, width = total_image.shape[:2]
    linesize = int(min(width, height)/8)

    padding = int(min(width, height)/8)


    flat = np.sum(edges, axis = 0)
    flat[0] = 0
    s = flat.shape[0]
    flat[s - 1] = 0

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
        print("Did not detect picture in picture")
        return

    cap = cv2.VideoCapture(args.srcpath)
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

    video = VideoFileClip(args.srcpath)
    boxvideo = video.fx(crop, x1 = xl, y1 = yl, x2 = xh, y2 = yh)

    boxvideo.write_videofile(box_path)

    def drawrect(frame):
        cv2.rectangle(frame, (xl, yl), (xh, yh), (0,0,0), -1)
        return frame

    outervideo = video.fl_image(drawrect)

    outervideo.write_videofile(outer_path)


if __name__ == "__main__":
    main()
