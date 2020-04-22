#!/usr/bin/env python3

import sys
import os
from moviepy.editor import *
import glob
import argparse
import numpy as np
from skimage import transform as tf
import math
from random import randrange

up_multiplier = 1.2
up_multiplier_large = 2
down_multiplier = 0.8
down_multiplier_large = 0.5

static_path = "../assets/static.mp4"
snow_path = "../assets/snow.mp4"

def trapzWarp(frame,cx,cy,ismask=False):
    """ Complicated function (will be latex packaged as a fx) """
    Y,X = frame.shape[:2]
    src = np.array([[0,0],[X,0],[X,Y],[0,Y]])
    dst = np.array([[cx*X,cy*Y],[(1-cx)*X,cy*Y],[X,Y],[0,Y]])
    tform = tf.ProjectiveTransform()
    tform.estimate(src,dst)
    im = tf.warp(frame, tform.inverse, output_shape=(Y,X))
    return im if ismask else (im*255).astype('uint8')


class Attack:
    def frame(self, up):
        # print(f"frame up={up}")
        # newvid = self.speed(up)
        if up is True:
            frame_rate_multiplier = up_multiplier_large
        else:
            frame_rate_multiplier = down_multiplier_large
        self.fps = self.fps / frame_rate_multiplier

        return self.vid


    def scale(self, up, noise=False):
        # print(f"scale in={up}, noise={noise}")

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
                newvid = newvid.margin(
                    left = math.ceil(abs(current_w - newvid.w) / 2),
                    right = math.floor(abs(current_w - newvid.w) / 2),
                    top = math.ceil(abs(current_h - newvid.h) / 2),
                    bottom = math.floor(abs(current_h - newvid.h) / 2),
                    opacity = 0.0
                )
                # TODO: chandni\
                static_clip = VideoFileClip(static_path).fx(vfx.loop,
                    duration=newvid.duration)
                    # static_clip = static_clip.crop(
                    #     width = current_w,
                    #     height = current_h
                    # )
                newvid = CompositeVideoClip([static_clip, newvid],
                    size=(current_w,current_h))
            else:
                newvid = newvid.margin(
                    left = math.ceil(abs(current_w - newvid.w) / 2),
                    right = math.floor(abs(current_w - newvid.w) / 2),
                    top = math.ceil(abs(current_h - newvid.h) / 2),
                    bottom = math.floor(abs(current_h - newvid.h) / 2)
                )

        return newvid


    def recolor(self, grey=False, dark=False):
        # print(f"recolor grey={grey} dark={dark}")
        if grey is True:
            newvid = self.vid.fx(vfx.blackwhite)
        else:
            newvid = self.vid.fx(vfx.colorx, down_multiplier_large)

        return newvid


    def rotate(self, deg):
        # print(f"rotate deg={deg}")
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
        # print(f"speed up={up}")
        if up == True:
            multiplier = up_multiplier_large
        else:
            multiplier = down_multiplier_large
        return self.vid.speedx(factor=multiplier)


    def projection(self):
        print("projection")
        fl_im = lambda frame : trapzWarp(frame,0.1,0.1)
        warped_vid = self.vid.fl_image(fl_im)
        return warped_vid


    def exact_match(self):
        print("exact_match")
        return self.vid


    def snowflakes(self):
        print("snowflakes")
        snowflakes = VideoFileClip(snow_path).fx(vfx.loop,
            duration=self.vid.duration)
        snowflakes = snowflakes.resize(newsize=(self.vid.w, self.vid.h))
        snowflakes = snowflakes.fx(vfx.mask_color, color=[0,0,255], thr=850)
        newvid = CompositeVideoClip([self.vid, snowflakes],
            size=(self.vid.w,self.vid.h))
        return newvid


    def scale_up(self):
        return self.scale(up=True)


    def scale_down_black(self):
        return self.scale(up=False,noise=False)


    def scale_down_noise(self):
        return self.scale(up=False, noise=True)


    def frame_rate_up(self):
        return self.frame(up=True)


    def frame_rate_down(self):
        return self.frame(up=False)


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
        return self.vid.fx(vfx.mirror_x)

    def pic_in_pic(self, large, small, duration):
        print("pic_in_pic")
        othervid = large.fx(vfx.loop, duration=duration)
        newvid = small.resize(0.35).fx(vfx.loop, duration=duration)
        newvid = CompositeVideoClip(
            [othervid, newvid.set_position(("right", "top"))],
            size=(large.w, large.h)
        )
        return newvid

    def pic_in_pic_small(self):
        return self.pic_in_pic(self.base_video, self.vid, self.vid.duration)

    def pic_in_pic_large(self):
        return self.pic_in_pic(self.vid, self.base_video, self.vid.duration)


    def speed_up(self):
        return self.speed(up=True)


    def speed_down(self):
        return self.speed(up=False)


    def insert_clip(self, video):
        nonpirated_length = math.floor(self.base_video.duration)
        pirated_length = video.duration
        split_point = randrange(8,nonpirated_length)
        start_clip = randrange(math.floor(pirated_length * 2 / 5), math.floor(pirated_length * 3 / 5))
        clip_length = randrange(3, 6)
        clip1 = self.base_video.subclip(t_start=split_point)
        clip2 = video.subclip(t_start=start_clip, t_end=start_clip + clip_length)
        clip3 = self.base_video.subclip(t_start=6, t_end=split_point)
        return concatenate_videoclips([clip1,clip2,clip3])


    def generate_all(self):
        for attack_function in self.attack_functions:
            self.vid = VideoFileClip(self.vidpath)
            filename = os.path.splitext(os.path.basename(self.vidpath))[0]
            vidname = filename + "_" + attack_function.__name__
            vidpath = os.path.join(self.destdir, vidname + ".mp4")
            self.fps = self.vid.fps
            video = attack_function()
            video.write_videofile(vidpath, fps=self.fps, audio=False, verbose=False)
            inserted_video = self.insert_clip(video)
            insertname = vidname + "_inserted"
            insertpath = os.path.join(self.destdir, insertname + ".mp4")
            inserted_video.write_videofile(insertpath, audio=False)
            del self.vid.reader
            del self.vid


    def __init__(self, vidpath, destdir, basepath):
        self.vidpath = vidpath
        self.destdir = destdir
        self.base_video = VideoFileClip(basepath)
        self.attack_functions = [
            # self.projection,
            # self.exact_match,
            # self.snowflakes,
            # self.scale_up,
            # self.scale_down_black,
            # self.scale_down_noise,
            # self.frame_rate_up,
            # self.frame_rate_down,
            # self.recolor_grey,
            # self.recolor_dark,
            # self.rotate_90,
            # self.rotate_180,
            # self.mirror,
            self.pic_in_pic_large,
            self.pic_in_pic_small,
            # self.speed_up,
            # self.speed_down
        ]


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

    fullvids = glob.glob(args.srcdir + "*")

    for vidpath in fullvids:
        Attacker = Attack(vidpath, args.destdir, args.extravid)
        Attacker.generate_all()


if __name__ == '__main__':
    main()
