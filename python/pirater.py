#!/usr/bin/env python3

import sys
import os
from moviepy.editor import *
import glob
import argparse
import numpy as np
from skimage import transform as tf
import math

up_multiplier = 1.2
down_multiplier = 0.8

class Attack:
    def frame(self, up):
        print(f"frame up={up}")
        newvid = self.speed(up)
        if up is True:
            frame_rate_multiplier = up_multiplier
        else:
            frame_rate_multiplier = down_multiplier
        print(newvid.fps)
        old_fps = newvid.fps
        newvid = newvid.set_fps(30 / frame_rate_multiplier)

        return newvid

    def scale(self, up, noise=False):
        print(f"scale in={up}, noise={noise}")

        current_w = self.vid.w
        current_h = self.vid.h

        if up is True:
            newvid = self.vid.resize(up_multiplier)
            x1 = math.ceil(abs(current_w - newvid.w) / 2)
            y1 = math.ceil(abs(current_h - newvid.h) / 2)
            newvid = newvid.crop(x1=x1,y1=y1,x2=current_w+x1,y2=current_h)
        else:
            newvid = self.vid.resize(down_multiplier)
            if noise is True:
                # TODO: chandni
                print("oops not done")
            else:
                newvid = newvid.margin(
                    left = math.ceil(abs(current_w - newvid.w) / 2),
                    right = math.floor(abs(current_w - newvid.w) / 2),
                    top = math.ceil(abs(current_h - newvid.h) / 2),
                    bottom = math.floor(abs(current_h - newvid.h) / 2)
                )

        return newvid

    def recolor(self, grey=False, dark=False):
        print(f"recolor grey={grey} dark={dark}")
        if grey is True:
            newvid = self.vid.blackwhite()
        else:
            newvid = self.vid.colorx(down_multiplier)

        return newvid

    def rotate(self, deg):
        print(f"rotate deg={deg}")
        current_h = self.vid.h
        current_w = self.vid.w
        newvid = self.vid.rotate(deg)
        if deg == 90:
            newvid = newvid.resize(self.vid.h / newvid.h)
            newvid = newvid.margin(
                left = math.ceil(abs(current_w - newvid.w) / 2),
                right = math.floor(abs(current_w - newvid.w) / 2),
                top = math.ceil(abs(current_h - newvid.h) / 2),
                bottom = math.floor(abs(current_h - newvid.h) / 2)
            )
        return newvid

    def speed(self, up):
        print(f"speed up={up}")
        if up == True:
            multiplier = up_multiplier
        else:
            multiplier = down_multiplier
        return self.vid.speedx(factor=multiplier)

    # https://zulko.github.io/moviepy/examples/star_worms.html
    def bottomWarp(self, pic):
        Y,X = pic.shape[:2]
        scale_factor = 0.1
        src = np.array([[0,0],[X,0],[X,Y],[0,Y]])
        dst = np.array([[0,0],[X,0],[X - scale_factor * X,Y],[0 + scale_factor * X,Y]])
        tform = tf.ProjectiveTransform()
        tform.estimate(src ,dst)
        im = tf.warp(pic, tform)
        return im

    def projection(self):
        warp_im = lambda pic : self.bottomWarp(pic)
        modified_clip= self.vid.fl_image(warp_im)
        print("projection")

        return modified_clip

    def exact_match(self):
        print("exact_match")
        return self.vid

    def snowflakes(self):
        print("snowflakes")
        return self.vid

    def scale_up(self):
        return self.scale(up=True)

    def scale_down_black(self):
        return self.scale(up=False,noise=False)

    def scale_down_noise(self):
        return self.scale(up=False, noise=True)

    def frame_rate_up(self):
        return self.vid.speedx(factor=2)

    def frame_rate_down(self):
        return self.vid.speedx(factor=0.5)

    def recolor_grey(self):
        return self.recolor(grey=True)

    def recolor_dark(self):
        return self.recolor(dark=True)

    def rotate_90(self):
        return self.rotate(deg=90)

    def rotate_180(self):
        return self.rotate(deg=180)

    def mirror(self):
        print("mirror")
        return self.vid.mirror_y()

    def pic_in_pic(self):
        print("pic_in_pic")
        return self.vid

    def speed_up(self):
        return self.speed(up=True)

    def speed_down(self):
        return self.speed(up=False)

    def insert_clip(self, video, base_video):
        return self.vid

    def generate_all(self):
        for attack_function in self.attack_functions:
            print(self.vidpath)
            filename = os.path.splitext(os.path.basename(self.vidpath))[0]
            print(filename)
            vidname = filename + "_" + attack_function.__name__
            vidpath = os.path.join(self.destdir, vidname + ".mp4")
            video = attack_function()
            video.write_videofile(vidpath, audio=False)
            inserted_video = self.insert_clip(video, self.base_video)
            insertname = vidname + "_inserted"
            insertpath = os.path.join(self.destdir, insertname + ".mp4")
            inserted_video.write_videofile(insertpath, audio=False)

    def __init__(self, vidpath, destdir, basepath):
        self.vidpath = vidpath
        self.vid = VideoFileClip(vidpath)
        self.destdir = destdir
        self.base_video = VideoFileClip(basepath)
        self.attack_functions = [
            # self.projection,
            # self.exact_match,
            # self.snowflakes,
            # self.scale_up,
            # self.scale_down_black,
            # self.scale_down_noise,
            self.frame_rate_up,
            self.frame_rate_down,
            # self.recolor_grey,
            # self.recolor_dark,
            # self.rotate_90,
            # self.rotate_180,
            # self.mirror,
            # self.pic_in_pic,
            # self.speed_up,
            # self.speed_down
        ]



# def generate_and_save_all_attacks(vidpath, destdir):
#     for attack_function in attack_functions:
#         video = attack_function(vidpath)
#         video.write_videofile(destdir, audio=False)

def main():
    parser = argparse.ArgumentParser(
        description='Generate testing videos with inserter clips.')

    # Positional arguments
    parser.add_argument(
        'srcdir',
        metavar='SOURCEDIR',
        type=str,
        help='path to directory of full videos to use as a base')

    parser.add_argument('destdir', metavar='DESTDIR', type=str,
                        help='path to directory to output testing videos')

    parser.add_argument('extravid', metavar='EXTRAVID', type=str,
                        help='path to extra video for insertion')


    args = parser.parse_args()

    fullvids = []
    fullvids.append(glob.glob(args.srcdir + "*")[0])

    for vidpath in fullvids:
        Attacker = Attack(vidpath, args.destdir, args.extravid)
        Attacker.generate_all()

if __name__ == '__main__':
    main()
