# python gen_testvids.py ../data/fullvids ../data/testingvids -n 10 -i 2


import sys
from moviepy.editor import *
from itertools import combinations
import glob
import random
import argparse


def insert(original_fname, pirated_fnames, outpath):
    """Insert a clip of pirated into original."""

    print('Inserting {0} clips into {1}'.format(len(pirated_fnames), original_fname))

    original = VideoFileClip(original_fname)
    pirated = list(map(VideoFileClip, pirated_fnames))
    clips = []

    pirated_duration = original.duration / len(pirated_fnames)

    start_original = 0
    for i in range(len(pirated)):
        # Play the original from where we left off
        end_original = start_original + pirated_duration
        if end_original > original.duration:
            end_original = original.duration
        original_clip = original.subclip(start_original, end_original)

        print('i: {0}, filename: {1}, start: {2}, end: {3}'.format(i, original.filename, start_original, end_original))
        start_original += (end_original - start_original)

        # Generate a random pirated clip
        start_pirated = random.randrange(int(pirated[i].duration))
        end_pirated = start_pirated + pirated_duration
        if end_pirated > pirated[i].duration:
            end_pirated = pirated[i].duration

        print('i: {0}, filename: {1}, start: {2}, end: {3}'.format(i, pirated[i].filename, start_pirated, end_pirated))
        pirated_clip = pirated[i].subclip(start_pirated, end_pirated)

        clips.append(original_clip)
        clips.append(pirated_clip)

    video = concatenate_videoclips(clips)

    # Export the video
    video.write_videofile(outpath, audio = False)


def main():
    parser = argparse.ArgumentParser(description =
        'Generate testing videos with inserter clips.')

    # Positional arguments
    parser.add_argument('srcdir', metavar = 'SOURCEDIR', type = str,
        help = 'path to directory of full videos to use as a base')
    parser.add_argument('destdir', metavar = 'DESTDIR', type = str,
        help = 'path to directory to output testing videos')

    # Optional arguments
    args = parser.add_argument('-n', type = int,
        help = 'number of testing videos to generate')
    args = parser.add_argument('-i', type = int,
        help = 'number of clips to insert per video')

    args = parser.parse_args()

    # Generate all possible combinations of two videos from database
    fullvids = glob.glob(args.srcdir + "*")
    sets = list(combinations(fullvids, args.i + 1))

    # Random sample of n for testing
    sets = random.sample(sets, args.n)

    i = 0
    for set in sets:
        outpath = args.destdir + "test" + str(i) + ".mp4"
        insert(set[0], set[1:], outpath)
        i += 1


if __name__ == "__main__":
    main()
