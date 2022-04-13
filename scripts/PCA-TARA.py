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
from sklearn.metrics import roc_curve, auc

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
    plt.savefig('images/SVM/confusion-matrix/TARA/PCA/'+filename)
    plt.close()

def plot_auc(y_pred, y_actual, title, filename):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    plt.savefig('images/SVM/AUC/TARA/PCA/'+filename)
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
data = pd.read_csv('files/data/condensedKO_features.csv', index_col=0)
labels = pd.read_csv('files/data/labels.csv', index_col=0)

#get ocean regions
int_labels = labels*1
del int_labels['site']
master_labels = int_labels.idxmax(axis=1)

sites = data['site']
#print(sites)

print('Splitting data...')
features = data.loc[:, ~data.columns.isin(['site', 'Station.label','Layer','polar','lower.size.fraction','upper.size.fraction','Event.date','Latitude','Longitude','Depth.nominal',
'Ocean.region','Temperature','Oxygen','ChlorophyllA','Carbon.total','Salinity','Gradient.Surface.temp(SST)','Fluorescence','CO3','HCO3','Density','PO4','PAR.PC','NO3','Si',
'Alkalinity.total','Ammonium.5m','Depth.Mixed.Layer','Lyapunov','NO2','Depth.Min.O2','NO2NO3','Nitracline','Brunt.Väisälä','Iron.5m','Depth.Max.O2','Okubo.Weiss','Residence.time'])]
features = pd.get_dummies(features)
#print(features)
labels = labels.loc[:, ~labels.columns.isin(['site'])]
#print(labels)

print('Cleaning features...')
#print([features[col].isna().sum() for col in features.columns if features[col].isna().sum() != 0])
remove = [col for col in features.columns if features[col].isna().sum() != 0 or col.__contains__('Ocean.region')]
# fill.remove('PAR.PC')
# features = features.loc[:, ~features.columns.isin(['PAR.PC'])] #remove columns with too many missing values
# features.fillna(method='')

features = features.loc[:, ~features.columns.isin(remove)] #remove columns with too many missing values

print('Performing PCA...')
#pca_model = PCA(n_components=0.99) #account for 99% of variability
pca_model = PCA(n_components=0.9) #make 9 components for 9 ocean regions
pca_features = pca_model.fit_transform(features)

#plot_pca(master_labels, pca_model, pca_features, 'images/PCA/nc_9.png')

pca_features_df = pd.DataFrame(pca_features)

sm = SMOTE(k_neighbors=1, random_state=55)
#features_res, labels_res = sm.fit_resample(features, labels)

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

print()
for label in labels.columns:
    X_res, y_res = sm.fit_resample(pca_features_df, labels[label])
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=5)#, shuffle=True, stratify=labels[label]) # 70% training and 30% test
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    #X_test_res, y_test_res = sm.fit_resample(X_test, y_test)

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
