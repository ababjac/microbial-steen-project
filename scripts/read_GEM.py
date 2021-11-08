import pandas as pd
import chardet

def detect_encoding(file):
    guess = chardet.detect(open(file, 'rb').read())['encoding']
    return guess


metadata = pd.read_csv('files/data/GEM_metadata.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/GEM_metadata.tsv'))
#print(metadata)
annot_features = pd.read_csv('files/data/annotation_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/annotation_features_counts_wide.tsv'))
#print(annot_features)
path_features = pd.read_csv('files/data/pathway_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/pathway_features_counts_wide.tsv'))
#print(path_features)

tmp = pd.merge(metadata, annot_features, on='genome_id', how='inner')
df = pd.merge(tmp, path_features, on='genome_id', how='inner')

print(df)
