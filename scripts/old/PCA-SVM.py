import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import plotly.express as px

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
    plt.savefig('images/confusion-matrix/PCA-SVM/'+filename)
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
features = data.loc[:, ~data.columns.isin(['site'])]
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

#pca_model = PCA(n_components=0.99) #account for 99% of variability
pca_model = PCA(n_components=9) #make 9 components for 9 ocean regions
pca_features = pca_model.fit_transform(features)

plot_pca(master_labels, pca_model, pca_features, 'images/PCA/nc_9.png')

#pca_features_df = pd.DataFrame(pca_features)
#print(pca_features_df)



# print()
# for label in labels.columns:
#     X_train, X_test, y_train, y_test = train_test_split(pca_features_df, labels[label], test_size=0.3,random_state=5) # 70% training and 30% test
#
#     print('Building model for label:', label)
#     clf = svm.SVC(kernel='linear')
#     clf.fit(X_train, y_train)
#
#     print('Predicting on test data for label:', label)
#     y_pred = clf.predict(X_test)
#
#     print('Calculating metrics for:', label)
#     print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#     print("Precision:",metrics.precision_score(y_test, y_pred))
#     print("Recall:",metrics.recall_score(y_test, y_pred))
#
#     print('Plotting:', label)
#     plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=label, filename=label+'_CM.png')
#
#     print()
