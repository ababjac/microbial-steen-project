import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as ks
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from keras.wrappers.scikit_learn import KerasClassifier

def scale(train, test):
    xtrain_scaled = pd.DataFrame(MinMaxScaler().fit_transform(train), columns=train.columns)
    xtest_scaled = pd.DataFrame(MinMaxScaler().fit_transform(test), columns=test.columns)
    return xtrain_scaled, xtest_scaled

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
    plt.savefig('images/SVM/confusion-matrix/TARA/AE/'+filename)
    plt.close()

def plot_auc(y_pred, y_actual, title, filename):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    plt.savefig('images/SVM/AUC/TARA/AE/'+filename)
    plt.close()

class Autoencoder(ks.models.Model):
    def __init__(self, actual_dim, latent_dim, activation, loss, optimizer):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = ks.Sequential([
        ks.layers.Flatten(),
        ks.layers.Dense(latent_dim, activation=activation),
        ])

        self.decoder = ks.Sequential([
        ks.layers.Dense(actual_dim, activation=activation),
        #ks.layers.Reshape((actual_dim, actual_dim))
        ])

        self.compile(loss=loss, optimizer=optimizer, metrics=[ks.metrics.BinaryAccuracy(name='accuracy')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_AE(actual_dim=1, latent_dim=100, activation='relu', loss='MAE', optimizer='Adam'):
    return Autoencoder(actual_dim, latent_dim, activation, loss, optimizer)


print('Reading data...')
data = pd.read_csv('files/data/condensedKO_features.csv', index_col=0)
labels = pd.read_csv('files/data/labels.csv', index_col=0)

#get ocean regions
int_labels = labels*1
del int_labels['site']
master_labels = int_labels.idxmax(axis=1)

sites = data['site']

print('Splitting data...')
#features = data.loc[:, ~data.columns.isin(['site'])]
features = data.loc[:, ~data.columns.isin(['site', 'Station.label','Layer','polar','lower.size.fraction','upper.size.fraction','Event.date','Latitude','Longitude','Depth.nominal',
'Ocean.region','Temperature','Oxygen','ChlorophyllA','Carbon.total','Salinity','Gradient.Surface.temp(SST)','Fluorescence','CO3','HCO3','Density','PO4','PAR.PC','NO3','Si',
'Alkalinity.total','Ammonium.5m','Depth.Mixed.Layer','Lyapunov','NO2','Depth.Min.O2','NO2NO3','Nitracline','Brunt.V??is??l??','Iron.5m','Depth.Max.O2','Okubo.Weiss','Residence.time'])]
features = pd.get_dummies(features)
labels = labels.loc[:, ~labels.columns.isin(['site'])]

print('Cleaning features...')
remove = [col for col in features.columns if features[col].isna().sum() != 0 or col.__contains__('Ocean.region')]
features = features.loc[:, ~features.columns.isin(remove)] #remove columns with too many missing values

sm = SMOTE(k_neighbors=1, random_state=55)

print()

params_SVM = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear'],
    'probability': [True]
}

clf = GridSearchCV(
    estimator=svm.SVC(),
    param_grid=params_SVM,
    cv=5,
    n_jobs=5,
    verbose=3
)

params_AE = {
    'actual_dim' : [len(features.columns)],
    'latent_dim' : [10, 50, 100, 200],
    'activation' : ['relu', 'sigmoid', 'tanh'],
    'loss' : ['MAE', 'binary_crossentropy'],
    'optimizer' : ['SGD', 'Adam']
}

model = KerasClassifier(build_fn=create_AE, epochs=10, verbose=0)
grid = GridSearchCV(
    estimator=model,
    param_grid=params_AE,
    cv=5,
#    n_jobs=3,
    verbose=3
)

#for label in labels.columns: #it doesnt seem to work in a for loop - strange
print()
for label in labels.columns:

    print('Scaling data...')
    X_res, y_res = sm.fit_resample(features, labels[label])
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=5)
    X_train_scaled, X_test_scaled = scale(X_train, X_test)

    print('Building autoencoder model...')
    result = grid.fit(X_train_scaled, X_train_scaled, validation_data=(X_test_scaled, X_test_scaled))
    params = grid.best_params_
    #print(params)

    autoencoder = create_AE(**params) #create autoencoder with best parameters from grid search

    try:
        encoder_layer = autoencoder.encoder
    except:
        exit

    AE_train = pd.DataFrame(encoder_layer.predict(X_train_scaled))
    AE_train.add_prefix('feature_')
    AE_test = pd.DataFrame(encoder_layer.predict(X_test_scaled))
    AE_test.add_prefix('feature_')

    #print(AE_train.shape)

    #AE_train = preprocessing.scale(AE_train)
    #AE_test = preprocessing.scale(AE_test)

    print('Predicting with SVM...')

    print('Building model for label:', label)
    clf.fit(AE_train, y_train)

    print('Predicting on test data for label:', label)
    y_pred = clf.predict(AE_test)
    y_prob = clf.predict_proba(AE_test) #get probabilities for AUC
    probs = y_prob[:,1]

    print('Calculating AUC score...')
    #plot_auc(probs, y_test, 'AUC for '+label, label+'_AUC-nometa.png')

    print('Calculating metrics for:', label)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

    print('Plotting:', label)
    #plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=label, filename=label+'_CM-nometa.png')

    print()
