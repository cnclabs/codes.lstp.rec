import numpy as np
import pandas as pd
import csv

train = {}

with open('data/beauty_train.txt', 'r') as f:
    rows = csv.reader(f)
    for row in rows:
        r = row[0].split()
        if r[0] not in train:
            train[r[0]] = []
            train[r[0]].append(r[1])
        else:
            train[r[0]].append(r[1])

#LSTP graph
for n in [5, 10, 15]:
    output = []
    for u, v in train.items():
        for i in v:
            output.append(['pseudo_'+u, i, '1'])
        for i in v[-n:]:
            output.append([u, i, '10'])
    with open("data/beauty_lstp_" + str(n) + ".txt", "w", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(output)
