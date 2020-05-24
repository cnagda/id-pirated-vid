import pygame as pg
import numpy as np
import time
from moviepy.editor import *

from moviepy.decorators import convert_masks_to_RGB, requires_duration
from moviepy.tools import cvsecs

from random import randrange

VIDNAME = 0
Q_START = 4
Q_END = 5
DB_START = 2
DB_END = 3
SCORE = 1

# Adapted from moviepy's preview()
class Previewer:

    def draw_button(self, text, center):
        mysmallfont = pg.font.SysFont('Arial', 20)
        button_loc = (center[0] - self.button_w / 2,
                center[1] - self.button_h/4,
                self.button_w,
                self.button_h)
        self.screen.fill(pg.Color("black"), button_loc)
        surf = mysmallfont.render(text, False, (0, 0, 0))
        loc = (center[0] - surf.get_size()[0] / 2, center[1])
        pg.draw.rect(self.screen, (randrange(180,220), randrange(180,220), randrange(180,220)), button_loc)
        self.screen.blit(surf, loc)
        pg.display.update()


    def init_screen(self):
        self.screen = pg.display.set_mode((self.viewer_w, self.viewer_h), 0)

        pg.font.init()
        mylargefont = pg.font.SysFont('Arial', 30)

        surf1 = mylargefont.render('Query video', False, (255, 255, 255))
        surf2 = mylargefont.render('Database video', False, (255, 255, 255))

        loc1 = ((self.margin + self.video_w / 2 - surf1.get_size()[0] / 2),(self.margin))
        loc2 = ((self.margin * 2 + self.video_w * 3 / 2 - surf2.get_size()[0] / 2),(self.margin))

        self.screen.blit(surf1, loc1)
        self.screen.blit(surf2, loc2)

        self.play_pause_loc = (self.viewer_w / 2, self.margin * 3 + self.video_h)

        self.button_w = self.margin
        self.button_h = self.margin / 2


        self.draw_button('PLAY', self.play_pause_loc)



    def imdisplay(self, imarray1, imarray2):
        # imarr1 on left, imarr2 on right
        a = pg.surfarray.make_surface(imarray1.swapaxes(0, 1))
        b = pg.surfarray.make_surface(imarray2.swapaxes(0, 1))
        self.screen.blit(a, (self.margin, self.margin * 2))
        self.screen.blit(b, (self.video_w + 2 * self.margin, self.margin *2))
        pg.display.flip()


    def check_events(self, center):
        clicked = False
        exit = False
        for event in pg.event.get():
            if event.type == pg.QUIT or (
                event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
            ):
                exit = True
            if event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if center[0] + self.button_w / 2 > event.pos[0] > center[0] - self.button_w / 2:
                        if center[1] + self.button_h / 2 > event.pos[1] > center[1] - self.button_h / 2:
                            clicked = True

        return (clicked, exit)

    def pause(self):
        (clicked, exit) = self.check_events(self.play_pause_loc)
        while clicked is False:
            if exit:
                return True
            (clicked, exit) = self.check_events(self.play_pause_loc)

        self.draw_button('PAUSE', self.play_pause_loc)
        return False

    def play_from_beginning(self, fps=10):

        img1 = self.queryvid.get_frame(0)
        img2 = self.dbvid.get_frame(0)
        self.imdisplay(img1, img2)

        t0 = time.time()

        for t in np.arange(1.0 / fps, max(self.queryvid.duration - 0.001, self.dbvid.duration - 0.001), 1.0 / fps):

            img1 = self.queryvid.get_frame(t)
            img2 = self.dbvid.get_frame(t)

            (clicked, exit) = self.check_events(self.play_pause_loc)

            if exit:
                return True

            if clicked:
                self.draw_button('PLAY', self.play_pause_loc)
                early_return = self.pause()
                if early_return:
                    return True

            t1 = time.time()
            time.sleep(max(0, t - (t1 - t0)))
            self.imdisplay(img1, img2)


        self.draw_button('PLAY', self.play_pause_loc)
        return False



    def view(self, fps=10):
        img1 = self.queryvid.get_frame(0)
        img2 = self.dbvid.get_frame(0)
        self.imdisplay(img1, img2)

        clicked = False
        while clicked is False:
            (clicked, exit) = self.check_events(self.play_pause_loc)
            if exit:
                return


        self.draw_button('PAUSE', self.play_pause_loc)

        early_return = self.play_from_beginning()

        while not early_return:
            (clicked, exit) = self.check_events(self.play_pause_loc)
            if exit:
                pg.display.quit()
                pg.quit()
                return
            if clicked:
                self.draw_button('PAUSE', self.play_pause_loc)
                early_return = self.play_from_beginning()

        pg.display.quit()
        pg.quit()


    def __init__(self, querypath, dbpath, row):
        self.video_w = 400
        self.margin = 50
        self.viewer_w = self.margin * 3 + self.video_w * 2
        self.queryvid = VideoFileClip(querypath).subclip(row[Q_START] / 1000., row[Q_END] / 1000.).resize(width=self.video_w)
        self.dbvid = VideoFileClip(dbpath).subclip(row[DB_START] / 1000., row[DB_END] / 1000. ).resize(width=self.video_w)
        self.video_h = max(self.queryvid.h, self.dbvid.h)
        self.viewer_h = self.video_h + self.margin * 4

        self.init_screen()
