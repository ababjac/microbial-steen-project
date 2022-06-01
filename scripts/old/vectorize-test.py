import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import plotly.express as px
import chardet
from statistics import mean

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
# print(list(metadata.columns))
# print(list(path_features.columns))

data = pd.merge(metadata, path_features, on='genome_id', how='inner')
#print(data)
#print(list(data.columns))
#print(data.shape)

#check for highly correlated path_features
# cor_matrix = data.corr().abs()
# upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
# to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
# print(to_drop) #empty list

#ids = data['genome_id']
#label_strings = data['cultured.status']

print('Vectorizing by family...')

#print(len(set(data['genus'])))
vectorized = pd.DataFrame()
for val in set(data['family']):
    sub = data[data['family'] == val]
    #print(sub)
    uncult = sub[sub['cultured.status'] == 'uncultured']

    if len(uncult) < 1:
        vectorized = vectorized.append(sub)
        continue

    for val2 in path_features.columns:
        if val2 == 'genome_id':
            continue

        avg = mean(uncult[val2])
        sub[val2] - avg

    #print(sub)
    vectorized = vectorized.append(sub)

#print(vectorized)

print('Splitting data...')
ids = vectorized['genome_id']
label_strings = vectorized['cultured.status']

features = vectorized.loc[:, ~vectorized.columns.isin(['genome_id', 'cultured.status'])]
features = pd.get_dummies(features)
#print(features)

labels = pd.get_dummies(label_strings)['cultured']
#print(labels)

print('Cleaning features...')
remove = [col for col in features.columns if features[col].isna().sum() != 0]
features = features.loc[:, ~features.columns.isin(remove)] #remove columns with too many missing values

print(features)
