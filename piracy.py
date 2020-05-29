#!/usr/bin/env python3
import argparse
import subprocess
import os


root_dir = os.path.abspath(os.path.dirname(__file__))
python_dir = os.path.join(root_dir, "python")
results_dir = os.path.join(root_dir, "results")
app_dir = root_dir
if not os.path.exists(os.path.join(app_dir, "add")):
    app_dir = os.path.join(root_dir, "build")
    if not os.path.exists(os.path.join(app_dir, "add")):
        raise Exception("can't find the build dir")

def validate_args(args):
    # TODO: add more validation

    # Validate database path exists
    # TODO: make sure database folder is properly formatted?
    if args.databasePath == "ADD":
        return True # creates db if DNE
    if os.path.isdir(os.path.join(os.getcwd(), args.databasePath)):
        return True
    print("Database path is invalid, exiting")
    return False

def expand_paths(paths):
    expanded = []
    for path in paths:
        if path=="-1":
            expanded.append(path)
        else:
            if os.path.isdir(path):
                for r,d,f in os.walk(path):
                    for file in f:
                        expanded.append(f"{r}/{file}")
            elif os.path.isfile(path):
                expanded.append(path)
            else:
                print(f"Invalid path: {path}, ignoring this one")
    return expanded


def call_add(args):
    for i, path in enumerate(args.paths):
        print('--------------------------------------------------------------')
        call_args = []

        call_args.append(os.path.join(app_dir, "add"))
        call_args.append(args.databasePath)
        call_args.append(path)
        kScene = "-1"
        kFrame = "-1"
        thresholdScene = "-1"
        if i == len(args.paths) - 1:
            kScene = str(args.kScene)
            kFrame = str(args.kFrame)
            thresholdScene = str(args.thresholdScene)
        call_args.append(kScene)
        call_args.append(kFrame)
        call_args.append(thresholdScene)

        print(f"Adding Video from Path: {path}")
        print(f"Options:   kFrame: {kFrame}   kScene: {kScene}   thresholdScene: {thresholdScene}")

        # if kFrame != "-1":
        #     print("Will reconstruct frame vocabulary")
        # if kScene != "-1":
        #     print("Will reconstruct scene vocabulary")
        # if thresholdScene != "-1":
        #     print("Will reconstruct scenes")

        subprocess.call(call_args)

def call_query(args):

    for i, path in enumerate(args.paths):
        print('--------------------------------------------------------------')
        call_args = []
        vidlist = []
        vidlist.append(path)
        if args.subimage is True:
            vidlist = []
            print("Looking for subimage\n")
            subprocess.call([
                'python3', os.path.join(python_dir, "subimage_detector.py"), path])

            outer_path = os.path.join(results_dir, "outervideo.mp4")
            box_path = os.path.join(results_dir, "boxvideo.mp4")

            if os.path.exists(outer_path) and os.path.exists(box_path):
                print("Will query twice (outervideo.mp4 and boxvideo.mp4)")
                vidlist.append(box_path)
                vidlist.append(outer_path)
            else:
                vidlist.append(path)

        matchlist = set()
        for vidpath in vidlist:
            call_args = []
            call_args.append(os.path.join(app_dir, "query"))
            call_args.append(args.databasePath)
            call_args.append(vidpath)

            if args.frames is True:
                call_args.append("--frames")

            print(f"Querying Video from Path: {vidpath}")

            # print(call_args)
            subprocess.call(call_args)


            logpath = os.path.join(results_dir, f"{os.path.basename(vidpath)}.csv")
            viewer_args = [os.path.join(root_dir, "viewer.py"),'-shortestmatch', str(args.shortestmatch), logpath, vidpath]
            if args.visualize:
                viewer_args.append('-v')
            subprocess.call(viewer_args)

            with open(os.path.join(results_dir, "resultcache.txt")) as file:
                matchlist.update(file.readlines())
        with open(os.path.join(results_dir, "resultcache.txt"), 'w') as file:
            for item in matchlist:
                file.write(f"{item}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Check videos for pirated content. '+
        'Use ADD to create or update a database. ' +
        'Use QUERY to check an existing database for one or more videos. ' +
        'Use INFO to see parameters for an existing database. ' +
        'For more information, try ADD, QUERY, or INFO with the -h option.'
    )

    '''
    Options:
    ADD [-kFrame] [-kScene] [-thresholdScene] path [path ...]
    QUERY databasePath path [path ...]
    '''

    subparsers = parser.add_subparsers(title='type', dest='type')
    subparsers.required = True

    parser_add = subparsers.add_parser('ADD')
    parser_add.add_argument(
        'databasePath',
        metavar='dbPath',
        help='path to database of known videos'
    )
    parser_add.add_argument(
        'paths',
        help='path(s) to directories/files to add',
        nargs='*',
        default=['-1']
    )
    parser_add.add_argument(
        '-kFrame',
        metavar='KF',
        help='k value for frame kmeans',
        default=-1,
        type=int
    )
    parser_add.add_argument(
        '-kScene',
        metavar='KS',
        help='k value for scene kmeans',
        default=-1,
        type=int
    )
    parser_add.add_argument(
        '-thresholdScene',
        metavar='TS',
        help='threshold for inter-scene similarity',
        default=-1,
        type=float
    )

    parser_query = subparsers.add_parser('QUERY')
    parser_query.add_argument(
        'databasePath',
        metavar='dbPath',
        help='path to database of known videos'
    )
    parser_query.add_argument(
        'paths',
        help='path(s) to directories/files to add',
        nargs='+'
    )
    parser_query.add_argument(
        '-v','--visualize',
        action='store_true',
        help='visualize video matches'
    )
    parser_query.add_argument(
        '-shortestmatch',
        metavar = 'SM',
        type=int,
        help='minimum length of matching video clip (in seconds)',
        default=0
    )
    parser_query.add_argument(
        '--frames',
        action='store_true',
        help='match frames instead of scenes; slower but more accurate'
    )
    parser_query.add_argument(
        '--subimage',
        action='store_true',
        help='additionally looks for a subimage that shares exactly one corner (such as a picture-in-picture attack)'
    )

    parser_info = subparsers.add_parser('INFO')
    parser_info.add_argument(
        'databasePath',
        metavar='dbPath',
        help='path to database of known videos'
    )

    args = parser.parse_args()
    # print(args)
    is_valid = validate_args(args)
    if is_valid:
        try:
            args.paths = expand_paths(args.paths)
        except:
            args.paths = []

        if args.type == "ADD":
            call_add(args)
        elif args.type == "QUERY":
            call_query(args)
        elif args.type == "INFO":
            subprocess.call([os.path.join(app_dir, "info"), args.databasePath])


if __name__ == "__main__":
    main()
