import pandas as pd
import chardet

def detect_encoding(file):
    guess = chardet.detect(open(file, 'rb').read())['encoding']
    return guess
    
metadata = pd.read_csv('files/data/rhizo_data/ITS_rhizosphere_metadata.csv', header=0, index_col=0, encoding=detect_encoding('files/data/rhizo_data/ITS_rhizosphere_metadata.csv'))
otu_features = pd.read_csv('files/data/rhizo_data/ITS_rhizosphere_otu.csv', header=0, index_col=0, encoding=detect_encoding('files/data/rhizo_data/ITS_rhizosphere_otu.csv'))
otu_T = otu_features.T

print(metadata.shape)
print(otu_T.shape)

data = pd.read_csv('files/data/condensedKO_features.csv', index_col=0)
labels = pd.read_csv('files/data/labels.csv', index_col=0)

print(data.shape)
print(list(data.columns))

metadata = pd.read_csv('files/data/GEM_metadata.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/GEM_metadata.tsv'))
annot_features = pd.read_csv('files/data/annotation_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/annotation_features_counts_wide.tsv'))
path_features = pd.read_csv('files/data/pathway_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/pathway_features_counts_wide.tsv'))

print(metadata.shape)
print(annot_features.shape)
print(path_features.shape)
