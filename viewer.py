#!/usr/bin/env python3

# import sys
import os
import glob
import argparse
# import vlc
import multiprocessing

import pandas as pd

import threading
from datetime import datetime
import datetime as dt

from enum import Enum

import csv


VIDNAME = 0
Q_START = 4
Q_END = 5
DB_START = 2
DB_END = 3
SCORE = 1

def read_logfile(logpath):
    # video_names = set()
    # reader = csv.reader(open(logpath, "r"), delimiter=",")
    # x = list(reader)[1:]
    # y = []
    # for row in x:
    #     r = []
    #     video_names.add(str(row[VIDNAME]))
    #     r.append(str(row[VIDNAME]))
    #     r.append(float(row[SCORE]))
    #     r.append(int(row[DB_START]))
    #     r.append(int(row[DB_END]))
    #     r.append(int(row[Q_START]))
    #     r.append(int(row[Q_END]))
    #     y.append(r)
    #
    # logfile = np.array(y,dtype='object')
    #
    # if len(logfile) > 1:
    #     logfile = logfile[np.argsort(logfile[:, SCORE])[::-1]]
    #     logfile = logfile[logfile[:,1] > np.mean(logfile[:,1]) + 1.5*np.std(logfile[:,1])]
    #

    logfile = pd.read_csv(logpath, index_col=None)

    df = logfile.groupby('Database Video').sum().reset_index()
    threshold = df['Confidence'].mean() + 1.5*df['Confidence'].std()
    df = df.loc[df['Confidence'] > threshold]
    df = df.sort_values('Confidence')
    video_names = df['Database Video'].tolist()

    logfile = logfile[logfile['Database Video'].isin(df['Database Video']).tolist()]
    logfile = logfile.to_numpy()

    with open("./results/resultcache.txt", "w") as file:
        for item in video_names:
            file.write(f"{item}\n")

    return logfile

def str_timestamp(num_ms):
    n_sec = int(dt.timedelta(milliseconds=num_ms).total_seconds())

    n_hrs = n_sec // 3600
    n_sec = n_sec % 3600

    n_min = n_sec // 60
    n_sec = n_sec % 60

    return "{:02d}:{:02d}:{:02d}".format(n_hrs, n_min, n_sec)


def print_log(logfile):
    if len(logfile) < 1:
        print("NO MATCHES FOUND")
        return
    print("{}:{:>30}{:>12}{:>30}{:>30}".format(
            "#", "Name of Matching Video","Score","Range in DB", "Range in Query"))
    for i, row in enumerate(logfile):

        db_range = "{} - {}".format(
                str_timestamp(row[DB_START]),
                str_timestamp(row[DB_END]))
        query_range = "{} - {}".format(
                str_timestamp(row[Q_START]),
                str_timestamp(row[Q_END]))
        print("{}:{:>30}{:>12}{:>30}{:>30}".format(
                i, row[VIDNAME],row[SCORE],db_range, query_range))


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
