from Bio import SeqIO
import pandas as pd
import os
import sys

DIR = sys.argv[1]
OUT = sys.argv[2]

records = []
with os.scandir(DIR) as d:

    for entry in d:
        if entry.name.endswith('.fna') or entry.name.endswith('.fa') and entry.is_file():
            input_path = os.path.join(DIR, entry.name)

            with open(input_path, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    records.append([entry.name, record.description, record.seq])

df = pd.DataFrame(records, columns=['filename', 'description', 'seq'])
df.to_csv(OUT)
