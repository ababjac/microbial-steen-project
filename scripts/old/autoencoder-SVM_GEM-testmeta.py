import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as ks
import chardet
import more_itertools
from imblearn.over_sampling import SMOTE

def scale(train, test):
    xtrain_scaled = pd.DataFrame(MinMaxScaler().fit_transform(train), columns=train.columns)
    xtest_scaled = pd.DataFrame(MinMaxScaler().fit_transform(test), columns=test.columns)
    return xtrain_scaled, xtest_scaled

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

def plot_confusion_matrix(y_pred, y_actual, title, filename):
    plt.gca().set_aspect('equal')
    cf_matrix = metrics.confusion_matrix(y_actual, y_pred)
    if len(cf_matrix) != 2: #if it predicts perfectly then confusion matrix returns incorrect form
        val = cf_matrix[0][0]
        tmp = [val, 0]
        cf_matrix = np.array([tmp, [0, 0]])

    #print(cf_matrix)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title(title+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    #print('saving image')
    plt.savefig('images/confusion-matrix/GEM/nometa/SMOTE/'+filename)
    plt.close()

class Autoencoder(ks.models.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = ks.Sequential([
        ks.layers.Flatten(),
        ks.layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = ks.Sequential([
        ks.layers.Dense(784, activation='sigmoid'),
        ks.layers.Reshape((28, 28))
        ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

def run_analyses(features, labels, remove_string):
    print('Cleaning features...')
    remove = [col for col in features.columns if features[col].isna().sum() != 0]
    features = features.loc[:, ~features.columns.isin(remove)] #remove columns with too many missing values
    #print(features)

    print()

    label = 'cultured'

    print('Scaling data...')
    sm = SMOTE(k_neighbors=1, random_state=55)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=5)
    X_test_res, y_test_res = sm.fit_resample(X_test, y_test)
    X_train_scaled, X_test_scaled = scale(X_train, X_test_res)

    print('Building autoencoder model...')
    autoencoder = Autoencoder(100)
    autoencoder.compile(loss='mae', optimizer='adam')

    try:
        encoder_layer = autoencoder.get_layer('sequential')
    except:
        exit

    AE_train = pd.DataFrame(encoder_layer.predict(X_train_scaled))
    AE_train.add_prefix('feature_')
    AE_test = pd.DataFrame(encoder_layer.predict(X_test_scaled))
    AE_test.add_prefix('feature_')
    #print(reduced_df)

    print('Predicting with SVM...')

    print('Building model for label:', label)
    #print(AE_train.shape())
    clf = svm.SVC(kernel='linear')
    clf.fit(AE_train, y_train)

    print('Predicting on test data for label:', label)
    y_pred = clf.predict(AE_test)

    print('Calculating metrics for:', label)
    print("Accuracy:",metrics.accuracy_score(y_test_res, y_pred))
    print("Precision:",metrics.precision_score(y_test_res, y_pred))
    print("Recall:",metrics.recall_score(y_test_res, y_pred))

    print('Plotting:', label)
    plot_confusion_matrix(y_pred=y_pred, y_actual=y_test_res, title=label, filename=label+'_CM-AE100-no'+remove_string+'.png')

    print()


print('Reading data...')
metadata = pd.read_csv('files/data/GEM_metadata.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/GEM_metadata.tsv'))
annot_features = pd.read_csv('files/data/annotation_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/annotation_features_counts_wide.tsv'))
annot_features = normalize_abundances(annot_features)
path_features = pd.read_csv('files/data/pathway_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/pathway_features_counts_wide.tsv'))
path_features = normalize_abundances(path_features)

data = pd.merge(metadata, path_features, on='genome_id', how='inner')
data = pd.merge(data, annot_features, on='genome_id', how='inner')

ids = data['genome_id']
label_strings = data['cultured.status']

print('Splitting data...')
#features = data.loc[:, ~data.columns.isin(['genome_id', 'cultured.status'])] #original way
features = data.loc[:, ~data.columns.isin(['genome_id', 'cultured.status', 'culture.level', 'taxonomic.dist', 'domain', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'completeness'])] #
#remove = ['culture.level', 'genome_length', 'completeness', 'domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
#combos = list(more_itertools.powerset(remove))
#print(len(combos))

features = pd.get_dummies(features)
labels = pd.get_dummies(label_strings)['cultured']

run_analyses(features, labels, 'culturelevel-notaxonomy-nocompleteness-includeannot')
# for l in combos:
#     s = ''
#     for elem in l:
#         s += '-no'+elem
#
#     new_features = features.loc[:, ~features.columns.isin(l)]
#     new_features = pd.get_dummies(new_features)
#
#     labels = pd.get_dummies(label_strings)['cultured']
#
#     run_analyses(new_features, labels, s)
