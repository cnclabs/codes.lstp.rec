import csv

user = []
item = []

with open('data/beauty_lstp_5.txt', 'r') as f:
    rows = csv.reader(f)
    for row in rows:
        r = row[0].split()
        user.append(r[0])
        item.append(r[1])

field = []
lstp_field = []
for u in set(user):
    lstp_field.append([u, 'u'])
    if u[:4] == 'user':
        field.append([u, 'u'])
for i in set(item):
    lstp_field.append([i, 'i'])
    field.append([i, 'i'])

with open("data/beauty_lstp_field.txt", "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(lstp_field)

with open("data/beauty_field.txt", "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(field)
