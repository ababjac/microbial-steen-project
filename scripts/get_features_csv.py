import pandas as pd
import numpy as np

print('Reading data...')
df = pd.read_csv('metatranscriptomes/salazar_profiles/OM_RGC_genus_KO_profiles_metat_rarefy.tsv')
meta = pd.read_csv('metatranscriptomes/salazar_profiles/salazar_metadata.csv', encoding='ISO-8859-1')
convert = pd.read_csv('metatranscriptomes/salazar_profiles/metat_to_metag.csv')

convert_df_to_meta = dict(zip(convert['metat'].values.tolist(), convert['site'].values.tolist())) #df_locations -> meta_locations
convert_meta_to_df = dict(zip(convert['site'].values.tolist(), convert['metat'].values.tolist())) #meta_locations -> df_locations

print('Data processing...')
exclude = [col for col in df.columns.tolist() if col not in convert['metat'].values.tolist()]
#exclude.remove('KO')
df = df.loc[:, ~df.columns.isin(exclude)]

#print('Condensing df by KO...')
#df = df.loc[:, ~df.columns.isin(exclude)]
#condensed = df.groupby('KO', as_index=True).sum()

norm_df = pd.DataFrame()
#normalize abundances
#for c in condensed.columns:
for c in df.columns:
    if c.__contains__('TARA'):
        #total = condensed.loc[:, c].sum()
        total = df.loc[:, c].sum()

        if total == 0: #skip because there is no point in predicting these sites
            continue

        norm_df[c] = df[c] / total
        #norm_df[c] = condensed[c] / total


df_transposed = norm_df.T
df_transposed.reset_index(inplace=True)
df_transposed = df_transposed.rename(columns = {'index':'site'})

for i in range(0, len(df_transposed)):
    df_transposed['site'][i] = convert_df_to_meta[df_transposed['site'][i]]

data = pd.merge(meta, df_transposed, on='site', how='inner')
#print(data)
data.to_csv('files/data/uncondensed_features.csv')
#data.to_csv('files/data/condensedKO_features.csv')
