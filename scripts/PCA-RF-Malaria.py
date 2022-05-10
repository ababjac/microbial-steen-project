import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import plotly.express as px
import chardet
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE

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
    plt.savefig('images/RF/confusion-matrix/malaria/PCA/'+filename)
    plt.close()

def plot_auc(y_pred, y_actual, title, filename):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    plt.savefig('images/RF/AUC/malaria/PCA/'+filename)
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
metadata1 = pd.read_csv('files/data/mok_meta.tsv', sep='\t', header=0, encoding=detect_encoding('files/data/mok_meta.tsv'))
metadata2 = pd.read_csv('files/data/zhu_meta.csv', header=0, encoding=detect_encoding('files/data/zhu_meta.csv'))
metadata1['SampleID'] = metadata1['SampleID'].str.replace('-', '.')
metadata = pd.merge(metadata1, metadata2, on='SampleID', how='inner')

expr_features = pd.read_csv('files/data/zhu_expr.txt', sep='\t', header=0, index_col=0, encoding=detect_encoding('files/data/zhu_expr.txt'))
expr_features = expr_features.T
expr_features = expr_features.reset_index()
expr_features.rename(columns={'index':'SampleID'}, inplace=True)

data = pd.merge(metadata, expr_features, on='SampleID', how='inner')

data = data[(data['Clearance'] >= 6) | (data['Clearance'] < 4)]
data['Resistant'] =  np.where(data['Clearance'] >= 6.0, 1, 0)
ids = data['SampleID']

print('Splitting data...')
labels = data['Resistant']


features = data.loc[:, ~data.columns.isin(['Clearance', 'Resistant', 'SampleID', 'GenotypeID', 'SampleID.Pf3k', 'Parasites clearance time', 'Field_site'])]
#features = features.loc[:, ~features.columns.isin(['FieldsiteName', 'Country', 'Hemoglobin(g/dL)', 'Hematocrit(%)', 'parasitemia', 'Parasite count', 'Sample collection time(24hr)', 'Patient temperature', 'Drug', 'ACT_partnerdrug', 'Duration of lag phase', 'PC50', 'PC90', 'Estimated HPI', 'Estimated gametocytes proportion', 'ArtRFounders', 'Timepoint', 'RNA', 'Asexual_stage', 'Lifestage', 'Long_class'])]
features = pd.get_dummies(features)

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
label = 'Resistant'

params = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy'],
}

clf = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=3
)

X_train, X_test, y_train, y_test = train_test_split(pca_features, labels, test_size=0.3, random_state=5, shuffle=True, stratify=labels) # 70% training and 30% test
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

print(y_test)

#sm = SMOTE(k_neighbors=3, random_state=555)
#X_test, y_test = sm.fit_resample(X_test, y_test)


print('Building model for label:', label)
clf.fit(X_train, y_train)

print('Predicting on test data for label:', label)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test) #get probabilities for AUC
probs = y_prob[:,1]

print('Calculating AUC score...')
plot_auc(probs, y_test, 'AUC for '+label, label+'_AUC-nosmote.png')

print('Calculating metrics for:', label)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

print('Plotting:', label)
plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=label, filename=label+'_CM-nosmote.png')

print()
