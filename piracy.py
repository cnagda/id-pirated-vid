#!/usr/bin/env python3
import argparse
import subprocess
import os

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

def call_execs(args):
    for path in args.paths:
        call_args = []
        if args.type == 'ADD':
            call_args.append('./build/add')
            call_args.append(args.databasePath)
            call_args.append(path)
            call_args.append(str(args.kScene))
            call_args.append(str(args.kFrame))
            call_args.append(str(args.thresholdScene))
        else:
            call_args.append('./build/query')
            call_args.append(args.databasePath)
            call_args.append(path)
        subprocess.call(call_args)


def main():
    parser = argparse.ArgumentParser(
        description='Check videos for pirated content'
    )

    '''
    Options:
    ADD [-kFrame] [-kScene] [-thresholdScene] path [path ...]
    QUERY databasePath path [path ...]
    '''

    subparsers = parser.add_subparsers(title='type', dest='type')

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
        type=int
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

    args = parser.parse_args()
    # print(args)
    is_valid = validate_args(args)
    if is_valid:
        args.paths = expand_paths(args.paths)
        call_execs(args)

if __name__ == "__main__":
    main()
