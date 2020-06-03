import pandas as pd
from datetime import datetime
import datetime as dt
import csv
import os


VIDNAME = 0
Q_START = 4
Q_END = 5
DB_START = 2
DB_END = 3
SCORE = 1
LENGTH = 6

WHITE = '\u001b[37m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
GREY = '\u001b[38;5;244m'
BOLD = '\u001b[1m'
UNDERLINE = '\u001b[4m'

def join_from_path(logpath1, logpath2, destdir):
    log1 = pd.read_csv(logpath1, index_col=None)
    log2 = pd.read_csv(logpath2, index_col=None)

    result = log1

    log1 = log1.reset_index()
    log1.columns.values[0] = 'id'
    log1['id'] = log1.index

    nonzero1 = log1.loc[log1['Confidence'] > 0]
    nonzero2 = log2.loc[log2['Confidence'] > 0]

    for index1, row1 in nonzero1.iterrows():
        for index2, row2 in nonzero2.iterrows():
            if ((row1['Database Video'] == row2['Database Video']) and ((row2['Start Time'] <= row1['Start Time'] < row2['End Time']) or (row1['Start Time'] <= row2['Start Time'] < row1['End Time']))):
                l1 = row1['End Time'] - row1['Start Time']
                l2 = row2['End Time'] - row2['Start Time']
                p1 = l1 / (l1 + l2)
                p2 = l2 / (l1 + l2)
                newscore = row1['Confidence']*p1 + row2['Confidence']*p2
                print('index: ' + str(index1) + ', score: ' + str(newscore))
                result.iloc[index1, result.columns.get_loc('Confidence')] = newscore
                result.iloc[index1, result.columns.get_loc('Start Time')] = min(row1['Start Time'], row2['Start Time'])
                result.iloc[index1, result.columns.get_loc('Query Start Time')] = min(row1['Query Start Time'], row2['Query Start Time'])
                result.iloc[index1, result.columns.get_loc('End Time')] = max(row1['End Time'], row2['End Time'])
                result.iloc[index1, result.columns.get_loc('Query End Time')] = max(row1['Query End Time'], row2['Query End Time'])

    # print(result)
    outpath = os.path.join(destdir, "combined.mp4.csv")
    result.to_csv(outpath, index=False)
    return outpath
    
def read_logfile(logpath, shortestmatch):
    logfile = pd.read_csv(logpath, index_col=None)

    df = logfile.groupby('Database Video').sum().reset_index()
    threshold = df['Confidence'].mean() + 1.5*df['Confidence'].std()
    df = df.loc[df['Confidence'] > threshold]
    df = df.sort_values('Confidence')
    video_names = df['Database Video'].tolist()

    # df['Length'] = df.apply(lambda row: float(row['End Time'] - row['Start Time']) / 1000, axis=1)
    df['Length'] = (df['End Time'] - df['Start Time']) / 1000.
    df = df.loc[df['Length'] > shortestmatch]


    # logfile = logfile[logfile['Database Video'].isin(df['Database Video']).tolist()]

    with open("./results/resultcache.txt", "w") as file:
        for item in video_names:
            file.write(f"{item}\n")

    return df.to_numpy()

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
    print(f"\n{BOLD}MATCH(ES) FOUND:{ENDC}\n")
    print("{}:{:>30}{:>12}{:>20}{:>24}{:>24}".format(
            "#", "Name of Matching Video","Score","Length (sec.)", "Range in DB", "Range in Query"))
    for i, row in enumerate(logfile):

        score = row[SCORE]
        score_color = RED
        if score > 70:
            score_color = GREEN
        elif score > 50:
            score_color = YELLOW

        db_range = "{} - {}".format(
                str_timestamp(row[DB_START]),
                str_timestamp(row[DB_END]))
        query_range = "{} - {}".format(
                str_timestamp(row[Q_START]),
                str_timestamp(row[Q_END]))
        print("{}:\033[94m{:>30}\033[0m{}{:>12}\033[0m\u001b[38;5;244m{:>20}{:>24}{:>24}\033[0m".format(
                i, row[VIDNAME],score_color,row[SCORE],row[LENGTH],db_range, query_range))
    print("\n")
