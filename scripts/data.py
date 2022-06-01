import pandas as pd
from sklearn.model_selection import train_test_split
import helpers

#--------------------------------------------------------------------------------------------------#

def read_data(features_path, labels_path):
    if features_path.__contains__('.csv'):
        features = pd.read_csv(features_path, header=0, encoding=helpers.detect_encoding(features_path))
    elif features_path.__contains__('.tsv'):
        features = pd.read_csv(features_path, sep='\t', header=0, encoding=helpers.detect_encoding(features_path))
    else:
        print('Incorrect format - Must be .csv or .tsv file')
        exit

    if labels_path.__contains__('.csv'):
        labels = pd.read_csv(labels_path, header=0, encoding=helpers.detect_encoding(labels_path))
    elif labels_path.__contains__('.tsv'):
        labels = pd.read_csv(labels_path, sep='\t', header=0, encoding=helpers.detect_encoding(labels_path))
    else:
        print('Incorrect format - Must be .csv or .tsv file')
        exit

    return features, labels

#--------------------------------------------------------------------------------------------------#

def clean_data(data):
    remove = [col for col in data.columns if data[col].isna().sum() != 0]
    return data.loc[:, ~data.columns.isin(remove)] #this gets rid of remaining NA

#--------------------------------------------------------------------------------------------------#

def remove_metadata(features, metadata_list):
    return features.loc[:, ~features.columns.isin(metadata_list)]

#--------------------------------------------------------------------------------------------------#

def split_and_scale_data(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=5)
    X_train_scaled, X_test_scaled = helpers.standard_scale(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

#--------------------------------------------------------------------------------------------------#

def parse_metadata_csv(path):
    file = open(path)
    text = file.read()
    return list(text.split(',')) #list of metadata names
