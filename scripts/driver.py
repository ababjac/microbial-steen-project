import pandas as pd
import numpy as np

import helpers
import data
import preload
import models
import classifiers
import analyze

import argparse

parser = argparse.ArgumentParser(description="Driver Code for Dimensionality Reduction Machine Learning Pipeline")
parser.add_argument("-m","--model", type=str, required=True, choices=['AE', 'LASSO', 'PCA', 'tSNE'], help="Dimensionality Reduction technique to use on given features.")
parser.add_argument("-c","--classifier", type=str, required=True, choices=['RF', 'SVM'], help="Classification technique to use on predicted labels.")
parser.add_argument("-p","--preload", default=None, type=str, choices=['GEM', 'MALARIA', 'RHIZO', 'TARA'], help="Optional: Dataset to preload. Note: Will not be used if features/labels files are passed.")
parser.add_argument("-f","--features_file", default=None, type=str, help="Optional: Path to file containing features data. Note: used only if preload not specified.")
parser.add_argument("-l","--labels_file", default=None, type=str, help="Optional: Path to file containing labels (must correspond to features). Note: Used only if preload not specified.")
parser.add_argument("-r","--metadata_file", default=None, type=str, help="Optional: Path to file containing metadata column names to be removed (expected csv format). Note: Only will be used if include_metadata set to False.")
parser.add_argument("-d","--image_directory", default=None, type=str, help="Optional: Path (relative or absolute) to directory that images should be saved in.")
parser.add_argument("-s","--SMOTE", default='False', type=str, choices=['True', 'False', 'true', 'false'], metavar="True/False", help="Optional: Perform SMOTE to rebalance data. Default False.")
parser.add_argument("-i","--include_metadata", default='True', type=str, choices=['True', 'False', 'true', 'false'], metavar="True/False", help="Optional: Include metadata features? Default False. If False, -r/--metadata_file must be passed.")

args = parser.parse_args()

bool_include_metadata = args.include_metadata.lower() == 'true'
bool_SMOTE = args.SMOTE.lower() == 'true'

print('Reading data...')

if args.preload != None: #use a preloaded dataset
    if args.preload == 'GEM':
        features, labels = preload.preload_GEM(include_metadata=bool_include_metadata)
    elif args.preload == 'TARA':
        features, labels = preload.preload_TARA(include_metadata=bool_include_metadata)
    elif args.preload == 'RHIZO':
        features, labels = preload.preload_RHIZO(include_metadata=bool_include_metadata)
    elif args.preload == 'MALARIA':
        features, labels = preload.preload_MALARIA(include_metadata=bool_include_metadata)


else: #read in from custom files
    if args.features_file == None or args.labels_file == None:
        print('Features and labels files required if not using preloaded data.')
        exit

    features, labels = data.read_data(args.features_file, args.labels_file)

    if not bool_include_metadata and args.metadata_file != None:
        remove = data.parse_metadata_csv(args.metadata_file)
        features = data.remove_metadata(features, remove)

if isinstance(labels, pd.Series): #then there is only one label of interest (for example GEM, RHIZO, MALARIA)
    image_name = args.model+'-'+args.classifier+'-'+labels.name
    if not bool_include_metadata:
        image_name += '-nometa'
    if bool_SMOTE:
        image_name += '-smote'

    analyze.run_analysis(features, labels, bool_SMOTE, args.model, args.classifier, image_name, image_path=args.image_directory)

else: #there are multiple labels of interest (for example TARA)
    for label in labels.columns:
        image_name = args.model+'-'+args.classifier+'-'+label
        if not bool_include_metadata:
            image_name += '-nometa'
        if bool_SMOTE:
            image_name += '-smote'

        analyze.run_analysis(features, labels[label], bool_SMOTE, args.model, args.classifier, image_name, image_path=args.image_directory)
