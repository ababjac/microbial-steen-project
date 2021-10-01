import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print('Reading data...')
df = pd.read_csv('metatranscriptomes/salazar_profiles/OM_RGC_genus_KO_profiles_metat_rarefy.tsv')
meta = pd.read_csv('metatranscriptomes/salazar_profiles/salazar_metadata.csv', encoding='ISO-8859-1')
convert = pd.read_csv('metatranscriptomes/salazar_profiles/metat_to_metag.csv')

convert_df_to_meta = dict(zip(convert['metat'].values.tolist(), convert['site'].values.tolist())) #df_locations -> meta_locations
convert_meta_to_df = dict(zip(convert['site'].values.tolist(), convert['metat'].values.tolist())) #meta_locations -> df_locations

print('Data processing...')
exclude = [col for col in df.columns.tolist() if col not in convert['metat'].values.tolist()]
df = df.loc[:, ~df.columns.isin(exclude)]

norm_df = pd.DataFrame()
#normalize abundances
for c in df.columns:
    if c.__contains__('TARA'):
        total = df.loc[:, c].sum()

        if total == 0: #skip because there is no point in predicting these sites
            continue

        norm_df[c] = df[c] / total


df_transposed = norm_df.T
df_transposed.reset_index(inplace=True)
df_transposed = df_transposed.rename(columns = {'index':'site'})

for i in range(0, len(df_transposed)):
    df_transposed['site'][i] = convert_df_to_meta[df_transposed['site'][i]]

data = pd.merge(meta, df_transposed, on='site', how='inner')
#print(data)

print('Splitting data...')
features = data.loc[:, ~data.columns.isin(['site'])]
features = pd.get_dummies(features)

labels = data.loc[:, 'site']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,random_state=5) # 70% training and 30% test

print('Building model...')
rf = RandomForestClassifier(n_estimators=100, random_state=5)
rf.fit(X_train, y_train)

print('Predicting on test data...')
y_pred = rf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

conf_mat = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
sns.heatmap(conf_mat, annot=True)
plt.show()

print('Finding important features...')
feature_imp = pd.Series(rf.feature_importances_,index=features.columns).sort_values(ascending=False)
