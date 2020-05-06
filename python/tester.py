#!/usr/bin/env python3
import subprocess
import argparse
import glob
import os
import pickle

# NOTE: MUST BE RUN FROM ROOT FOLDER
# Also, for this to work properly for now, DB names can't have underscores
# Attack video naming scheme: NameInDB_kindofattack.*

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
        vidpaths = glob.glob(args.srcdir + "*")

    # Results in dictionary
    results = dict()

    # Empty dump file at beginning
    with open('./results/dump.txt', 'w') as f2:
        f2.write("")

    for vidpath in vidpaths:
        # print(f"Querying: {vidpath}")
        vidname = os.path.basename(vidpath)

        # Run query
        output = subprocess.check_output(['python3', 'piracy.py', 'QUERY', args.dbpath, vidpath])

        # Check if result matches expected
        result = ""
        with open("./results/resultcache.txt") as file:
            result = file.read()
        success = 0
        outstr = "Failure"
        # Name is assumed to contain a single extension ie .mpg
        pirated_from = vidname.split("_")[0]
        if result.split(".")[0] == pirated_from:
            success = 1
            outstr = "Success"
        # print(outstr)

        # record 0 for failure, 1 for success
        results[vidname] = success

        with open('./results/dump.txt', 'a') as f2:
            f2.write(f"{vidname}\n{output}\n{outstr}\n")

    with open('./results/allresults.txt','w') as f:
        f.write(str(results))

    with open("./results/allresults.pkl", 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    print(results)

if __name__ == '__main__':
    main()
