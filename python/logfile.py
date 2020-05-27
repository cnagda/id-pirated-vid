import pandas as pd
from datetime import datetime
import datetime as dt
import csv


VIDNAME = 0
Q_START = 4
Q_END = 5
DB_START = 2
DB_END = 3
SCORE = 1

def read_logfile(logpath, shortestmatch):
    logfile = pd.read_csv(logpath, index_col=None)

    df = logfile.groupby('Database Video').sum().reset_index()
    threshold = df['Confidence'].mean() + 1.5*df['Confidence'].std()
    df = df.loc[df['Confidence'] > threshold]
    df = df.sort_values('Confidence')
    video_names = df['Database Video'].tolist()

    df['Length'] = df.apply(lambda row: float(row['End Time'] - row['Start Time']) / 1000, axis=1)
    df = df.loc[df['Length'] > shortestmatch]

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
