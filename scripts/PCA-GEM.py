import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import plotly.express as px
import chardet
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrix(y_pred, y_actual, title, filename):
    cf_matrix = metrics.confusion_matrix(y_actual, y_pred)
    if len(cf_matrix) != 2: #if it predicts perfectly then confusion matrix returns incorrect form
        val = cf_matrix[0][0]
        tmp = [val, 0]
        cf_matrix = np.array([tmp, [0, 0]])

    #print(cf_matrix)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Greens')

    ax.set_title(title+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.savefig('images/confusion-matrix/GEM/PCA/'+filename)
    plt.close()

def plot_auc(y_pred, y_actual, title, filename):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    plt.savefig('images/AUC/GEM/PCA/'+filename)
    plt.close()

def plot_pca(colors, pca, components, filename, num):

    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    print(labels)
    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(num),
        color=colors
    )

    fig.update_traces(diagonal_visible=False)
    fig.write_image(filename)

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
annot_features = pd.read_csv('files/data/annotation_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/annotation_features_counts_wide.tsv'))
annot_features = normalize_abundances(annot_features)
path_features = pd.read_csv('files/data/pathway_features_counts_wide.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/pathway_features_counts_wide.tsv'))
path_features = normalize_abundances(path_features)

data = pd.merge(metadata, path_features, on='genome_id', how='inner')
data = pd.merge(data, annot_features, on='genome_id', how='inner')

ids = data['genome_id']
label_strings = data['cultured.status']

print('Splitting data...')
features = data.loc[:, ~data.columns.isin(['genome_id', 'cultured.status'])]
features = pd.get_dummies(features)

labels = pd.get_dummies(label_strings)['cultured']

print('Cleaning features...')
remove = [col for col in features.columns if features[col].isna().sum() != 0]
features = features.loc[:, ~features.columns.isin(remove)] #remove columns with missing values
features = features.values


print('Running PCA...')
pca_model = PCA(n_components=0.9) #Whatever is necessary to capture 90% of variability
pca_features = pca_model.fit_transform(features)


print()

#predict using SVM
print('Predicting with SVM...')
label = 'cultured'

params = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear'],
    'probability': [True]
}

clf = GridSearchCV(
    estimator=svm.SVC(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=3
)

X_train, X_test, y_train, y_test = train_test_split(pca_features, labels, test_size=0.3, random_state=5, shuffle=True, stratify=labels) # 70% training and 30% test

print('Building model for label:', label)
clf.fit(X_train, y_train)

print('Predicting on test data for label:', label)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test) #get probabilities for AUC
preds = y_prob[:,1]

print('Calculating AUC score...')
plot_auc(preds, y_test, 'AUC for '+label, label+'_AUC-nometa.png')

print('Calculating metrics for:', label)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

print('Plotting:', label)
plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=label, filename=label+'_CM-PCA.png')

print()
