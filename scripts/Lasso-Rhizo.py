import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import seaborn as sns
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
import chardet

def plot_confusion_matrix(y_pred, y_actual, title, filename):
    plt.gca().set_aspect('equal')
    cf_matrix = metrics.confusion_matrix(y_actual, y_pred)
    if len(cf_matrix) != 2: #if it predicts perfectly then confusion matrix returns incorrect form
        val = cf_matrix[0][0]
        tmp = [val, 0]
        cf_matrix = np.array([tmp, [0, 0]])

    #print(cf_matrix)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Reds')

    ax.set_title(title+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    plt.savefig('images/confusion-matrix/Rhizo/Lasso/'+filename)
    plt.close()

def detect_encoding(file):
    guess = chardet.detect(open(file, 'rb').read())['encoding']
    return guess

def scale(train, test):
    xtrain_scaled = pd.DataFrame(StandardScaler().fit_transform(train), columns=train.columns)
    xtest_scaled = pd.DataFrame(StandardScaler().fit_transform(test), columns=test.columns)
    return xtrain_scaled, xtest_scaled

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

print('Scaling data...')
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=5)
X_train_scaled, X_test_scaled = scale(X_train, X_test)

print('Doing feature selection with Lasso...')
pipeline = Pipeline([('scaler',StandardScaler()), ('model',Lasso())])
search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )
search.fit(X_train, y_train)
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
#print(list(importance))
remove = np.array(features.columns)[importance == 0]

if len(remove) < len(features.columns): #if everything is not important then no feature selection can occur
    X_train = X_train.loc[:, ~X_train.columns.isin(remove)]
    X_test = X_test.loc[:, ~X_test.columns.isin(remove)]

print('Predicting with SVM...')

params = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear']
}

clf = GridSearchCV(
    estimator=svm.SVC(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=3
)

print('Building model for label:', label)
clf.fit(X_train, y_train)

print('Predicting on test data for label:', label)
y_pred = clf.predict(X_test)

print('Calculating metrics for:', label)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

print('Plotting:', label)
plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=label, filename=label+'_CM-nometa.png')

print()
