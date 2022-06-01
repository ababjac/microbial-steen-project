import pandas as pd
import chardet

def detect_encoding(file):
    guess = chardet.detect(open(file, 'rb').read())['encoding']
    return guess

annot_features = pd.read_csv('files/data/annotation_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/annotation_features_counts_wide.tsv'))
annot_T = annot_features.T
annot_T.to_csv('files/data/annotation_features_counts_wide_transform.csv')
