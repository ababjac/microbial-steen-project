import os

def parse_file(path):
    file = open(path)
    memory = 0.0

    for line in file.readlines()[1:]:
        memory += float(line.split(' ')[1])

    return memory

DIR = 'files/memory/'
OUT = open(DIR+'stats.txt', 'w')

with os.scandir(DIR) as d:
    for entry in d:
        if entry.name.endswith('.dat') and entry.is_file():
            path = os.path.join(DIR, entry.name)

            model = entry.name.partition('.')[0]
            memory = parse_file(path)

            OUT.write(model+': '+str(memory)+'\n')
