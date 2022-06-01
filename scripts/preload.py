import pandas as pd
import numpy as np
import os
import helpers

#--------------------------------------------------------------------------------------------------#

def preload_GEM(include_metadata=True):
    curr_dir = os.getcwd()

    meta_file = curr_dir+'/files/data/GEM_data/GEM_metadata.tsv'
    annot_file = curr_dir+'/files/data/GEM_data/annotation_features_counts_wide.tsv'
    path_file = curr_dir+'/files/data/GEM_data/pathway_features_counts_wide.tsv'

    metadata = pd.read_csv(meta_file, sep='\t', header=0, encoding=helpers.detect_encoding(meta_file))
    annot_features = pd.read_csv(annot_file, sep='\t', header=0, encoding=helpers.detect_encoding(annot_file))
    annot_features = helpers.normalize_abundances(annot_features)
    path_features = pd.read_csv(path_file, sep='\t', header=0, encoding=helpers.detect_encoding(path_file))
    path_features = helpers.normalize_abundances(path_features)

    data = pd.merge(metadata, path_features, on='genome_id', how='inner')
    data = pd.merge(data, annot_features, on='genome_id', how='inner')

    label_strings = data['cultured.status']

    features = data.loc[:, ~data.columns.isin(['genome_id', 'cultured.status'])] #remove labels
    if not include_metadata: #remove metadata
        features = features.loc[:, ~features.columns.isin(['culture.level',
                                                           'taxonomic.dist',
                                                           'domain',
                                                           'phylum',
                                                           'class',
                                                           'order',
                                                           'family',
                                                           'genus',
                                                           'species',
                                                           'completeness'
                                                           ])]

    features = pd.get_dummies(features)
    labels = pd.get_dummies(label_strings)['cultured']

    return features, labels

#--------------------------------------------------------------------------------------------------#

def preload_TARA(include_metadata=True):
    curr_dir = os.getcwd()

    data_file = curr_dir+'/files/data/TARA_data/condensedKO_features.csv'
    labels_file = curr_dir+'/files/data/TARA_data/labels.csv'

    data = pd.read_csv(data_file, index_col=0, encoding=helpers.detect_encoding(data_file))
    labels = pd.read_csv(labels_file, index_col=0, encoding=helpers.detect_encoding(labels_file))

    features = data.loc[:, ~data.columns.isin(['site', 'Station.label', 'Event.date', 'Latitude', 'Longitude', 'Ocean.region'])] #remove labels
    if not include_metadata: #remove metadata
        features = features.loc[:, ~features.columns.isin(['Layer',
                                                           'polar',
                                                           'lower.size.fraction',
                                                           'upper.size.fraction',
                                                           'Depth.nominal',
                                                           'Ocean.region',
                                                           'Temperature',
                                                           'Oxygen',
                                                           'ChlorophyllA',
                                                           'Carbon.total',
                                                           'Salinity',
                                                           'Gradient.Surface.temp(SST)',
                                                           'Fluorescence',
                                                           'CO3',
                                                           'HCO3',
                                                           'Density',
                                                           'PO4',
                                                           'PAR.PC',
                                                           'NO3',
                                                           'Si',
                                                           'Alkalinity.total',
                                                           'Ammonium.5m',
                                                           'Depth.Mixed.Layer',
                                                           'Lyapunov',
                                                           'NO2',
                                                           'Depth.Min.O2',
                                                           'NO2NO3',
                                                           'Nitracline',
                                                           'Brunt.Väisälä',
                                                           'Iron.5m',
                                                           'Depth.Max.O2',
                                                           'Okubo.Weiss',
                                                           'Residence.time'
                                                           ])]

    features = pd.get_dummies(features)
    labels = labels.loc[:, ~labels.columns.isin(['site'])]

    return features, labels

#--------------------------------------------------------------------------------------------------#

def preload_RHIZO(include_metadata=True):
    curr_dir = os.getcwd()

    meta_file = curr_dir+'/files/data/rhizo_data/ITS_rhizosphere_metadata.csv'
    otu_file = curr_dir+'/files/data/rhizo_data/ITS_rhizosphere_otu.csv'

    metadata = pd.read_csv(meta_file, header=0, index_col=0, encoding=helpers.detect_encoding(meta_file))
    otu_features = pd.read_csv(otu_file, header=0, index_col=0, encoding=helpers.detect_encoding(otu_file))
    otu_T = otu_features.T

    data = metadata.join(otu_T)

    label_strings = data['drought_tolerance']

    features = data.loc[:, ~data.columns.isin(['drought_tolerance', 'marker_gene', 'habitat'])] #remove labels
    if not include_metadata: #remove metadata
        features = features.loc[:, ~features.columns.isin(['irrigation'])]

    features = pd.get_dummies(features)
    labels = pd.get_dummies(label_strings)['HI30']

    return features, labels

#--------------------------------------------------------------------------------------------------#

def preload_MALARIA(include_metadata=True):
    curr_dir = os.getcwd()

    meta1_file = curr_dir+'/files/data/malaria_data/mok_meta.tsv'
    meta2_file = curr_dir+'/files/data/malaria_data/zhu_meta.csv'
    expr_file = curr_dir+'/files/data/malaria_data/new_expression.csv'

    metadata1 = pd.read_csv(meta1_file, sep='\t', header=0, encoding=helpers.detect_encoding(meta1_file))
    metadata2 = pd.read_csv(meta2_file, header=0, encoding=helpers.detect_encoding(meta2_file))
    metadata1['SampleID'] = metadata1['SampleID'].str.replace('-', '.')
    metadata = pd.merge(metadata1, metadata2, on='SampleID', how='inner')

    expr_features = pd.read_csv(expr_file, header=0, index_col=0, encoding=helpers.detect_encoding(expr_file)).fillna(0)
    expr_features = expr_features.T
    expr_features = expr_features.reset_index()
    expr_features.rename(columns={'index':'GenotypeID'}, inplace=True)

    data = pd.merge(metadata, expr_features, on='GenotypeID', how='inner')

    data = data[(data['Clearance'] >= 6) | (data['Clearance'] < 5)] #these are likely "semi-resistant" samples so remove
    data['Resistant'] =  np.where(data['Clearance'] >= 6.0, 1, 0)

    features = data.loc[:, ~data.columns.isin(['Clearance', 'Resistant', 'SampleID', 'GenotypeID', 'SampleID.Pf3k', 'Parasites clearance time', 'Field_site'])] #remove labels
    if not include_metadata: #remove metadata
        features = features.loc[:, ~features.columns.isin(['FieldsiteName',
                                                           'Country',
                                                           'Hemoglobin(g/dL)',
                                                           'Hematocrit(%)',
                                                           'parasitemia',
                                                           'Parasite count',
                                                           'Sample collection time(24hr)',
                                                           'Patient temperature',
                                                           'Drug',
                                                           'ACT_partnerdrug',
                                                           'Duration of lag phase',
                                                           'PC50',
                                                           'PC90',
                                                           'Estimated HPI',
                                                           'Estimated gametocytes proportion',
                                                           'ArtRFounders',
                                                           'Timepoint',
                                                           'RNA',
                                                           'Asexual_stage',
                                                           'Lifestage',
                                                           'Long_class'
                                                           ])]

    features = pd.get_dummies(features)
    labels = data['Resistant']

    return features, labels
