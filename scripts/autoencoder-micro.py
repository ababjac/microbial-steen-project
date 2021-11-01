import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from tensorflow import keras as ks

def declare_autoencoder(num_features, latent_space = 50):
    print("Instantiate Autoencoder...")
    autoencoder = ks.Sequential([
        ks.layers.Dense(num_features,input_shape=(num_features,)),
        ks.layers.Dense(latent_space,activation='relu'),
        ks.layers.Dense(num_features,activation='sigmoid')
    ])

    autoencoder.summary()

    opt = ks.optimizers.Adam(learning_rate=0.1)
    autoencoder.compile(optimizer=opt,loss='binary_crossentropy')

    return autoencoder

print('Reading data...')
data = pd.read_csv('files/data/condensedKO_features.csv', index_col=0)
labels = pd.read_csv('files/data/labels.csv', index_col=0)

#get ocean regions
int_labels = labels*1
del int_labels['site']
master_labels = int_labels.idxmax(axis=1)

sites = data['site']

print('Splitting data...')
features = data.loc[:, ~data.columns.isin(['site'])]
features = pd.get_dummies(features)

print('Cleaning features...')
remove = [col for col in features.columns if features[col].isna().sum() != 0 or col.__contains__('Ocean.region')]
features = features.loc[:, ~features.columns.isin(remove)] #remove columns with too many missing values

print('Training model...')
AE = declare_autoencoder(num_features=len(features.columns), latent_space = 9)
AE.fit(features, features, epochs=5, batch_size=1)
