import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import plotly.express as px
from imblearn.over_sampling import SMOTE
import chardet
from sklearn.metrics import roc_curve, auc

def detect_encoding(file):
    guess = chardet.detect(open(file, 'rb').read())['encoding']
    return guess

def plot_confusion_matrix(y_pred, y_actual, title, filename):
    plt.gca().set_aspect('equal')
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
    plt.savefig('images/SVM/confusion-matrix/Rhizo/PCA/'+filename)
    plt.close()

def plot_auc(y_pred, y_actual, title, filename):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    plt.savefig('images/SVM/AUC/Rhizo/PCA/'+filename)
    plt.close()

def plot_pca(colors, pca, components, filename):

    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    print(labels)
    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(9),
        color=colors
    )

    fig.update_traces(diagonal_visible=False)
    fig.write_image(filename)


print('Reading data...')
metadata = pd.read_csv('files/data/rhizo_data/ITS_rhizosphere_metadata.csv', header=0, index_col=0, encoding=detect_encoding('files/data/rhizo_data/ITS_rhizosphere_metadata.csv'))
otu_features = pd.read_csv('files/data/rhizo_data/ITS_rhizosphere_otu.csv', header=0, index_col=0, encoding=detect_encoding('files/data/rhizo_data/ITS_rhizosphere_otu.csv'))
otu_T = otu_features.T

data = metadata.join(otu_T)
#print(data)

ids = data.index.values.tolist()
label_strings = data['drought_tolerance']

print('Splitting data...')
features = data.loc[:, ~data.columns.isin(['drought_tolerance', 'marker_gene', 'irrigation', 'habitat'])] #get rid of labels
features = pd.get_dummies(features)
#print(features)

labels = pd.get_dummies(label_strings)['HI30']
#print(labels)

print('Cleaning features...')
remove = [col for col in features.columns if features[col].isna().sum() != 0]
features = features.loc[:, ~features.columns.isin(remove)] #remove columns with too many missing values
#print(features)

print()

label = 'drought_tolerance'

print('Running PCA...')
pca_model = PCA(n_components=0.9) #make 9 components for 9 ocean regions
pca_features = pca_model.fit_transform(features)

#plot_pca(master_labels, pca_model, pca_features, 'images/PCA/nc_9.png')

pca_features_df = pd.DataFrame(pca_features)

#sm = SMOTE(k_neighbors=1, random_state=55)

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
#print(clf.best_params_)

print('Running SVM...')

#X_res, y_res = sm.fit_resample(pca_features_df, labels)
X_train, X_test, y_train, y_test = train_test_split(pca_features_df, labels, test_size=0.3, random_state=5)#, shuffle=True, stratify=labels[label]) # 70% training and 30% test
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

print('Building model for label:', label)
clf.fit(X_train, y_train)

print('Predicting on test data for label:', label)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test) #get probabilities for AUC
probs = y_prob[:,1]

print('Calculating AUC score...')
plot_auc(probs, y_test, 'AUC for '+label, label+'_AUC-nometa.png')

print('Calculating metrics for:', label)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

print('Plotting:', label)
plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=label, filename=label+'_CM-nometa.png')

print()
