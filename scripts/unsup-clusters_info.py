import pandas as pd
import numpy as np

print('Reading data...')
df = pd.read_csv('metatranscriptomes/salazar_profiles/OM_RGC_genus_KO_profiles_metat_rarefy.tsv')
meta = pd.read_csv('metatranscriptomes/salazar_profiles/salazar_metadata.csv', encoding='ISO-8859-1')
convert = pd.read_csv('metatranscriptomes/salazar_profiles/metat_to_metag.csv')

convert_df_to_meta = dict(zip(convert['metat'].values.tolist(), convert['site'].values.tolist())) #df_locations -> meta_locations
convert_meta_to_df = dict(zip(convert['site'].values.tolist(), convert['metat'].values.tolist())) #meta_locations -> df_locations

#figure out which columns do not have metadata and exclude
exclude = [col for col in df.columns.tolist() if col not in convert['metat'].values.tolist()]
exclude.remove('KO')
#print(exclude)

# print('Factoring out sites based on depth...')
# meta['Depth.nominal'].replace('nan', np.float64('NaN'))

print('Condensing df by KO...')
df = df.loc[:, ~df.columns.isin(exclude)]
condensed = df.groupby('KO', as_index=False).sum()

print('Parsing cluster file...')
file = open('files/clusters/unsupervised/full.txt')
cluster_text = file.readlines()

clusters = []
for line in cluster_text:
    if line.__contains__('Cluster') or line.__contains__('Singletons'):
        continue

    clust = line.split(', ')
    clust.remove('\n')

    if not clust:
        continue

    clusters.append(clust)

#print(clusters)

con_transposed = condensed.set_index('KO')
con_transposed = con_transposed.T
con_transposed.reset_index(inplace=True)

sites = []
for c in clusters:
    l = []
    for ko in c:
        #[con_transposed['index'][i] for i in range(len(con_transposed[ko])) if con_transposed[ko][i] != 0]
        for i in range(len(con_transposed[ko])):
            if con_transposed[ko][i] != 0:
                l.append(con_transposed['index'][i])
        #print(l)

    sites.append(list(set(l)))

clust_lengths = [len(l) for l in clusters]
print(clust_lengths)

site_lengths = [len(l) for l in sites]
print(site_lengths)

# print('Data processing...')
# exclude = [col for col in df.columns.tolist() if col not in convert['metat'].values.tolist()]
# df = df.loc[:, ~df.columns.isin(exclude)]
#
# norm_df = pd.DataFrame()
# #normalize abundances
# for c in df.columns:
#     if c.__contains__('TARA'):
#         total = df.loc[:, c].sum()
#
#         if total == 0: #skip because there is no point in predicting these sites
#             continue
#
#         norm_df[c] = df[c] / total
#
#
# df_transposed = norm_df.T
# df_transposed.reset_index(inplace=True)
# df_transposed = df_transposed.rename(columns = {'index':'site'})
#
# for i in range(0, len(df_transposed)):
#     df_transposed['site'][i] = convert_df_to_meta[df_transposed['site'][i]]
#
# data = pd.merge(meta, df_transposed, on='site', how='inner')
