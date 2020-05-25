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
    python_folder = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.normpath(os.path.join(python_folder, ".."))
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
    with open(os.path.join(root_dir, "results", "dump.txt"), 'w') as f2:
        f2.write("")


    options = [ name for name in os.listdir(args.dbpath) if os.path.isdir(os.path.join(args.dbpath, name)) ]

    for vidpath in vidpaths:
        vidname = os.path.basename(vidpath)
        print(f"Querying: {vidpath}")

        # Run query
        output = subprocess.check_output(['python3', os.path.join(root_dir, 'piracy.py'), 'QUERY', args.dbpath, vidpath])

        # Check if result matches expected
        with open(os.path.join(root_dir, "results", "resultcache.txt")) as file:
            vidlist = file.readlines()
        success = 0
        outstr = "Failure"
        # Name is assumed to contain a single extension ie .mpg
        pirated_from = vidname.split("_")[0]
        for result in vidlist:
            if result.split(".")[0] == pirated_from:
                success = 1
                outstr = "Success"

        # Not meant to be in database
        # if len(vidlist) == 0:
        found = False
        for vid in options:
            if pirated_from in vid:
                found = True
        if not found:
            vidname = f"{pirated_from}_not_in_db.mp4"
            if len(vidlist) == 0:
                success = 1
                outstr = "Success"

        # print(outstr)

        # record 0 for failure, 1 for success
        results[vidname] = success

        with open(os.path.join(root_dir, "results", "dump.txt"), 'a') as f2:
            f2.write(f"{vidname}\n{output}\n{outstr}\n")

    with open(os.path.join(root_dir, "results", "allresults.txt"),'w') as f:
        f.write(str(results))

    with open(os.path.join(root_dir, "results", "allresults.pkl"), 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    print(results)

if __name__ == '__main__':
    main()
