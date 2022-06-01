import os

def parse_file(path):
    file = open(path)
    mem = 0.0
    prev_mem = 0.0
    alloc_mem = 0.0

    for line in file.readlines()[1:]:
        mem = float(line.split(' ')[1])

        if mem > prev_mem:
            alloc_mem += mem-prev_mem

        prev_mem = mem

    return alloc_mem

DIR = 'files/memory/'
OUT = open(DIR+'stats.txt', 'w')

with os.scandir(DIR) as d:
    for entry in d:
        if entry.name.endswith('.dat') and entry.is_file():
            path = os.path.join(DIR, entry.name)

            model = entry.name.partition('.')[0]
            memory = parse_file(path)

            OUT.write(model+': '+str(memory)+'\n')
