#!/usr/bin/env python3
import subprocess
import argparse
import glob
import os
import pickle
from attack_vid_labels import *

# NOTE: MUST BE RUN FROM ROOT FOLDER
def main():

    # Read command-line args
    parser = argparse.ArgumentParser(
        description='Test attack videos with premade database')

    # Positional arguments
    parser.add_argument(
        'srcdir',
        metavar='SOURCEDIR',
        type=str,
        help='path to directory of attack videos')

    parser.add_argument('dbpath', metavar='DBPATH', type=str,
                        help='path to directory to output testing videos')

    args = parser.parse_args()


    # Create list of videopaths corresponding to videos to query
    vidpaths = []
    if os.path.isfile(args.srcdir):
        vidpaths.append(args.srcdir)
    elif os.path.isdir(args.srcdir):
        vidpaths = (glob.glob(args.srcdir + "*")[:2])

    # Results in dictionary
    results = dict()


    for vidpath in vidpaths:
        print(f"Querying: {vidpath}")
        vidname = os.path.basename(vidpath)

        # Run query
        subprocess.check_output(['./piracy.py', 'QUERY', args.dbpath, vidpath])

        # Check if result matches expected
        result = ""
        with open("resultcache.txt") as file:
            result = file.read()
        success = 0
        if result == AV_LABELS[vidname]:
            success = 1
            print ("Success")
        else:
            print("Failure")

        # record 0 for failure, 1 for success
        results[vidname] = success

    with open('allresults.txt','w') as f:
        f.write(str(results))

    with open("allresults.pkl", 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
