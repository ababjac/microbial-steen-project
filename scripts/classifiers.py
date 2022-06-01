import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import svm

import graphing

#--------------------------------------------------------------------------------------------------#

def run_SVM(X_train, X_test, y_train, y_test, image_name, image_path=None, param_grid=None, label=None, title=None, color=None):

        if param_grid == None:
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf', 'linear'],
                'probability': [True]
            }

        if label == None:
            label = y_train.name

        clf = GridSearchCV(
            estimator=svm.SVC(),
            param_grid=param_grid,
            cv=5,
            n_jobs=5,
            verbose=3
        )

        print('Building model for label:', label)
        clf.fit(X_train, y_train)

        print('Predicting on test data for label:', label)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test) #get probabilities for AUC
        probs = y_prob[:,1]

        print('Calculating metrics for:', label)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))

        if image_path != None:
            if image_path[-1] != '/':
                filename = image_path+'/'+image_name
            else:
                filename = image_path+image_name
        else:
            filename = image_name #save in current directory

        if title == None:
            title = label

        print('Calculating AUC score...')
        graphing.plot_auc(y_pred=probs, y_actual=y_test, title='AUC for '+title, path=filename+'_AUC.png')

        print('Plotting:', label)
        graphing.plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=title, path=filename+'_CM.png', color=color)

        print()

#--------------------------------------------------------------------------------------------------#

def run_RF(X_train, X_test, y_train, y_test, image_name, image_path=None, param_grid=None, label=None, title=None, color=None):

    if param_grid == None:
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            'criterion' :['gini', 'entropy'],
        }

    if label == None:
        label = y_train.name

    clf = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=5,
        n_jobs=5,
        verbose=3
    )

    print('Building model for label:', label)
    clf.fit(X_train, y_train)

    print('Predicting on test data for label:', label)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) #get probabilities for AUC
    probs = y_prob[:,1]

    print('Calculating metrics for:', label)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    if image_path != None:
        if image_path[-1] != '/':
            filename = image_path+'/'+image_name
        else:
            filename = image_path+image_name
    else:
        filename = image_name #save in current directory

    if title == None:
        title = label

    print('Calculating AUC score...')
    graphing.plot_auc(y_pred=probs, y_actual=y_test, title='AUC for '+title, path=filename+'_AUC.png')

    print('Plotting:', label)
    graphing.plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=title, path=filename+'_CM.png', color=color)

    print()
