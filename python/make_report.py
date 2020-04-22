import pickle
from collections import defaultdict
import re

# results = pickle.load(open("../../id-pirated-vid-db/results/allresults.pkl", "rb"))
results = pickle.load(open("../results/allresults.pkl", "rb"))

data = defaultdict(lambda: defaultdict(lambda: 0))
for fname in results:
    attack = "_".join(re.split("_|\\.", fname)[1:-1])
    if results[fname]:
        data[attack]['correct'] += 1
    data[attack]['total'] += 1

print("\t{:<16}\t{}\t{}".format("TYPE OF ATTACK", "FULL-LENGTH", "INSERTED CLIP"))
t_corr = 0
t_total = 0
t_corr_ins = 0
t_total_ins = 0
for attack in sorted(data):
    if "inserted" in attack:
        continue

    inserted = f"{attack}_inserted"

    t_corr += data[attack]['correct']
    t_total += data[attack]['total']
    t_corr_ins += data[inserted]['correct']
    t_total_ins += data[inserted]['total']


    s1 = "\t{:<16}\t".format(attack)
    s2 = "{}/{}\t\t".format(data[attack]['correct'], data[attack]['total'])
    s3 = "{}/{}".format(data[inserted]['correct'], data[inserted]['total'])
    print(s1+s2+s3)

# prevent divide by zero, 0/0 is 100%
if t_total_ins == 0:
    t_total_ins = 1
    t_corr_ins = 1

print("\t{:<16}\t{:.3f}%\t\t{:.3f}%".format("TOTAL", t_corr / t_total * 100, t_corr_ins / t_total_ins * 100))
