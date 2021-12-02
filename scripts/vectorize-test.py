import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import plotly.express as px
import chardet

def detect_encoding(file):
    guess = chardet.detect(open(file, 'rb').read())['encoding']
    return guess

def normalize_abundances(df):
    norm_df = pd.DataFrame()

    #normalize abundances
    for c in df.columns:
        if not c.__contains__('genome_id'):
            #total = condensed.loc[:, c].sum()
            total = df.loc[:, c].sum()

            if total == 0: #skip because there is no point in predicting these sites
                continue

            norm_df[c] = df[c] / total

    norm_df['genome_id'] = df['genome_id']
    return norm_df

print('Reading data...')
metadata = pd.read_csv('files/data/GEM_metadata.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/GEM_metadata.tsv'))
#annot_features = pd.read_csv('files/data/annotation_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/annotation_features_counts_wide.tsv'))
#annot_features = normalize_abundances(annot_features)
path_features = pd.read_csv('files/data/pathway_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/pathway_features_counts_wide.tsv'))
path_features = normalize_abundances(path_features)

data = pd.merge(metadata, path_features, on='genome_id', how='inner')

ids = data['genome_id']
label_strings = data['cultured.status']

print('Vectorizing...')
#deleted everything not working :(

print('Splitting data...')
features = data.loc[:, ~data.columns.isin(['genome_id', 'cultured.status'])]
features = pd.get_dummies(features)
#print(features)

labels = pd.get_dummies(label_strings)['cultured']
#print(labels)

print('Cleaning features...')
remove = [col for col in features.columns if features[col].isna().sum() != 0]
features = features.loc[:, ~features.columns.isin(remove)] #remove columns with too many missing values
