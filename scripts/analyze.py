import models
import classifiers
import data
import helpers

#--------------------------------------------------------------------------------------------------#

def run_analysis(features, labels, SMOTE, model, classifier, image_name, image_path=None, title=None, label=None, param_grid=None):
    print('Preparing data...')
    features = data.clean_data(features)
    X_train_scaled, X_test_scaled, y_train, y_test = data.split_and_scale_data(features, labels)

    if SMOTE:
        X_test_scaled, y_test = helpers.perform_SMOTE(X_test_scaled, y_test)


    print('Performing specified dimensionality reduction...')
    if model == 'AE':
        X_train_reduced, X_test_reduced = models.run_AE(X_train_scaled, X_test_scaled)
        color = 'Blues'
    elif model == 'LASSO':
        X_train_reduced, X_test_reduced = models.run_LASSO(X_train_scaled, X_test_scaled, y_train)
        color = 'Reds'
    elif model == 'PCA':
        X_train_reduced, X_test_reduced = models.run_PCA(X_train_scaled, X_test_scaled)
        color = 'Greens'
    elif model == 'tSNE':
        X_train_reduced, X_test_reduced = models.run_tSNE(X_train_scaled, X_test_scaled)
        color = 'Purples'
    else:
        print('Not a valid dimensionality reduction model')
        exit


    print('Performing specified classification task...')
    if classifier == 'SVM':
        classifiers.run_SVM(X_train_reduced, X_test_reduced, y_train, y_test, image_name, image_path=image_path, color=color)
    elif classifier == 'RF':
        classifiers.run_RF(X_train_reduced, X_test_reduced, y_train, y_test, image_name, image_path=image_path, color=color)
    else:
        print('Not a valid classifier')
        exit
