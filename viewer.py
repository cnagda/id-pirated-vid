#!/usr/bin/env python3

# import sys
import os
import glob
import argparse
# import vlc
# import multiprocessing
# import threading
# from enum import Enum
import python.logfile
from python.logfile import *


def my_glob(path):
    expanded = []
    for r,d,f in os.walk(path):
        for file in f:
            expanded.append(f"{r}/{file}")
    return expanded

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    parser = argparse.ArgumentParser(
        description='View results of query')

    # Positional arguments
    parser.add_argument(
        'logfile',
        type=str,
        help='path to result logfile')

    parser.add_argument(
        'querypath',
        type=str,
        help='path to query video')

    parser.add_argument(
        '-v','--visualize',
        action='store_true',
        help='visualize video matches'
    )

    args = parser.parse_args()

    if not os.path.isfile(args.logfile):
        print("Logfile does not exist at this path")
        return

    logfile = read_logfile(args.logfile)
    print_log(logfile)
    if not args.visualize or len(logfile) < 1:
        return

    from python.previewer import Previewer
    db_video_path = input("Please enter directory of videos in database to view alignments: ")
    while not os.path.isdir(db_video_path):
        print("Not a valid directory, please try again.")
        db_video_path = input("Please enter directory of videos in database to view alignments: ")


    vids = my_glob(db_video_path)

    # dict of vidname to path
    db_vids = {os.path.basename(vid) :vid for vid in vids}


    selection=""
    while (1):
        selection = input("Enter a number to view match, q to quit: ")
        if selection in ["q","Q","quit","Quit"]:
            break
        else:
            try:
                selection = int(selection)
                selected_row = logfile[selection]
            except:
                print("Invalid choice, try again")
                continue

        if selected_row[VIDNAME] in db_vids:
            previewer = Previewer(args.querypath, db_vids[selected_row[VIDNAME]], selected_row)
            previewer.view()
        else:
            print(f"{selected_row[VIDNAME]} is not in the database video directory")



if __name__ == '__main__':
    main()
