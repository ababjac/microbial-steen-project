import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#-----------------------------------------------------------------------------------------------------------#

def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=5)
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split - test_split) * len(df))]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds

#-----------------------------------------------------------------------------------------------------------#

print('Reading data...')
df = pd.read_csv('metatranscriptomes/salazar_profiles/OM_RGC_genus_KO_profiles_metat_rarefy.tsv')
meta = pd.read_csv('metatranscriptomes/salazar_profiles/salazar_metadata.csv', encoding='ISO-8859-1')
convert = pd.read_csv('metatranscriptomes/salazar_profiles/metat_to_metag.csv')
