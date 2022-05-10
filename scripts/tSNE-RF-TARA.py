import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import seaborn as sns
from sklearn import svm, metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
import chardet
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

def plot_confusion_matrix(y_pred, y_actual, title, filename):
    plt.gca().set_aspect('equal')
    cf_matrix = metrics.confusion_matrix(y_actual, y_pred)
    if len(cf_matrix) != 2: #if it predicts perfectly then confusion matrix returns incorrect form
        val = cf_matrix[0][0]
        tmp = [val, 0]
        cf_matrix = np.array([tmp, [0, 0]])

    #print(cf_matrix)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Purples')

    ax.set_title(title+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    plt.savefig('images/RF/confusion-matrix/TARA/tSNE/'+filename)
    plt.close()

def plot_auc(y_pred, y_actual, title, filename):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    plt.savefig('images/RF/AUC/TARA/tSNE/'+filename)
    plt.close()

def detect_encoding(file):
    guess = chardet.detect(open(file, 'rb').read())['encoding']
    return guess

def scale(train, test):
    xtrain_scaled = pd.DataFrame(StandardScaler().fit_transform(train), columns=train.columns)
    xtest_scaled = pd.DataFrame(StandardScaler().fit_transform(test), columns=test.columns)
    return xtrain_scaled, xtest_scaled

print('Reading data...')
data = pd.read_csv('files/data/condensedKO_features.csv', index_col=0)
labels = pd.read_csv('files/data/labels.csv', index_col=0)

#get ocean regions
int_labels = labels*1
del int_labels['site']
master_labels = int_labels.idxmax(axis=1)

sites = data['site']

print('Splitting data...')
features = data.loc[:, ~data.columns.isin(['site'])] #get rid of labels
#features = data.loc[:, ~data.columns.isin(['site', 'Station.label','Layer','polar','lower.size.fraction','upper.size.fraction','Event.date','Latitude','Longitude','Depth.nominal',
#'Ocean.region','Temperature','Oxygen','ChlorophyllA','Carbon.total','Salinity','Gradient.Surface.temp(SST)','Fluorescence','CO3','HCO3','Density','PO4','PAR.PC','NO3','Si',
#'Alkalinity.total','Ammonium.5m','Depth.Mixed.Layer','Lyapunov','NO2','Depth.Min.O2','NO2NO3','Nitracline','Brunt.Väisälä','Iron.5m','Depth.Max.O2','Okubo.Weiss','Residence.time'])]
features = pd.get_dummies(features)
#print(features)

labels = labels.loc[:, ~labels.columns.isin(['site'])]
#print(labels)

print('Cleaning features...')
remove = [col for col in features.columns if features[col].isna().sum() != 0 or col.__contains__('Ocean.region')]
features = features.loc[:, ~features.columns.isin(remove)] #remove columns with too many missing values
#print(features)

sm = SMOTE(k_neighbors=1, random_state=55)

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

params_tsne = {
    'n_components' : [2, 10, 20, 50],
    'perplexity' : [10, 20, 30, 40, 50],
    'learning_rate' : ['auto']
}

for label in labels.columns:

    print('Scaling data...')
    X_res, y_res = sm.fit_resample(features, labels[label])
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=5)


    print('Doing feature selection with t-SNE...')
    # pipeline = Pipeline([('scaler',StandardScaler()), ('model',TSNE())])
    # search = GridSearchCV(pipeline,
    #                     param_grid=params_tsne,
    #                     cv = 5,
    #                     verbose=3
    #                     )
    # search.fit(X_train, y_train)
    # tsne = TSNE(**search.best_params) #make best model
    # X_train = tsne.fit_transform(X_train)
    # X_test = tsne.fit_transform(X_test)
    tsne = TSNE(n_components=2)
    X_train = tsne.fit_transform(StandardScaler().fit_transform(X_train))
    X_test = tsne.fit_transform(StandardScaler().fit_transform(X_test))


    print('Predicting with SVM...')

    print('Building model for label:', label)
    clf.fit(X_train, y_train)

    print('Predicting on test data for label:', label)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) #get probabilities for AUC
    probs = y_prob[:,1]

    print('Calculating AUC score...')
    plot_auc(probs, y_test, 'AUC for '+label, label+'_AUC-nc2.png')

    print('Calculating metrics for:', label)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

    print('Plotting:', label)
    plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=label, filename=label+'_CM-nc2.png')

    print()
