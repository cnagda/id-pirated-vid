import pickle
from collections import defaultdict
import re

results = pickle.load(open("../../id-pirated-vid-db/results/allresults.pkl", "rb"))

data = defaultdict(lambda: defaultdict(lambda: 0))
for fname in results:
    attack = "_".join(re.split("_|\\.", fname)[1:-1])
    if results[fname]:
        data[attack]['correct'] += 1
    data[attack]['total'] += 1

for attack in sorted(data):
    s = "{}: {}/{}".format(attack, data[attack]['correct'], data[attack]['total'])
    print(s)
