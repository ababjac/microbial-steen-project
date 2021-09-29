import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cut_tree, fcluster, set_link_color_palette, dendrogram
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import itertools

print('Reading data...')
df = pd.read_csv('metatranscriptomes/salazar_profiles/OM_RGC_genus_KO_profiles_metat_rarefy.tsv')
meta = pd.read_csv('metatranscriptomes/salazar_profiles/salazar_metadata.csv', encoding='ISO-8859-1')
convert = pd.read_csv('metatranscriptomes/salazar_profiles/metat_to_metag.csv')

#df_subset = df.sample(n=5000, random_state=5) #for testing purposes use fewer sample

# for elem in df.columns.tolist():#meta['site']:
#     print(elem in convert['site'].values.tolist())

#print(df.columns)

#figure out which columns do not have metadata and exclude
exclude = [col for col in df.columns.tolist() if col not in convert['metat'].values.tolist()]
exclude.remove('KO')
#print(exclude)

# print('Factoring out sites based on depth...')
# meta['Depth.nominal'].replace('nan', np.float64('NaN'))

print('Condensing df by KO...')
df = df.loc[:, ~df.columns.isin(exclude)]
condensed = df.groupby('KO', as_index=False).sum()
#print(condensed)

#normalize by totals from columns
print('Normalizing df...')
norm_df = pd.DataFrame()

for c in condensed.columns:
    if c.__contains__('TARA'):
        total = condensed.loc[:, c].sum()

        if total == 0: #skip because there is no point in predicting these sites
            continue

        norm_df[c] = condensed[c] / total

#print(norm_df)


#calculate upper triangular distance matrix
print('Calculating pdist matrix...')
matrix = pdist(norm_df, 'braycurtis')
matrix_new = np.nan_to_num(matrix)

#calculate linkages
print('Calculating linkages...')
Z = linkage(matrix_new, 'ward')

######################################################################
#PREVIOUS CODE
######################################################################

# print('Picking threshold with CH_index')

# if FILENAME.__contains__('HD'):
#     thresholds = np.arange(0.005, 0.1, 0.005)
# elif FILENAME.__contains__('KS'):
#     thresholds = np.arange(0.1, 0.5, 0.01)

# thresholds = np.arange(0.02, 0.2, 0.005)
#
# errors = []
# for t in thresholds:
#     #print(t)
#     clusters = fcluster(Z, t, criterion='distance')
#     #print(len(set(clusters)))
#     errors.append(metrics.calinski_harabasz_score(matrix, list(clusters)))
#
# THRESHOLD = thresholds[errors.index(max(errors))]

# print('Plotting Elbow...')
# #Plot the elbow
# plt.plot(thresholds, errors, 'bx-')
# plt.xlabel('Threshold Value')
# plt.ylabel('CH Index')
# plt.title('Optimal Threshold Elbow Plot')
# plt.savefig(ERRORPLOT)
# plt.close()

#############################################################################

print('Plotting Dendrogram...')
colors = cm.gist_ncar(np.arange(0, 1, 0.1))

colorlst=[]# empty list where you will put your colors
for i in range(len(colors)): #get for your color hex instead of rgb
    colorlst.append(col.to_hex(colors[i]))

set_link_color_palette(colorlst)

D = dendrogram(Z, above_threshold_color='gray')
#plt.axhline(y=THRESHOLD, c='gray', lw=1, linestyle='dashed')

plt.title('Unsupervised Clusters using metatranscriptome abundances')

plt.savefig('images/dendrograms/fullplot.png')
plt.close()

#############################################################################

# print('Creating clusters using fcluster...')
# clusters = fcluster(Z, THRESHOLD, criterion='distance')

print('Creating clusers using cut_tree...')
clusters = cut_tree(Z, n_clusters=len(norm_df.columns.tolist()))
clusters = list(itertools.chain(*clusters.tolist()))
#print(clusters)

train['cluster_id'] = clusters

# data = {'Gene' : genes, 'cluster_id' : list(clusters)}
# df = pd.DataFrame(data)
length = len(set(clusters))

#OUT = OUTFILE+'_t'+str(round(THRESHOLD, 3))+'.txt'
OUT = 'files/clusters/unsupervised/full.txt'

clusters = []
singleton_cluster = []
for i in range(1, length+1):
    l = train['KO'][train['cluster_id'] == i].values.tolist()

    if len(l) == 1:
        singleton_cluster.append(l[0])
    else:
        clusters.append(l)

print('Writing to', OUT)
file = open(OUT, 'w')
for i in range(1, len(clusters)+1):
    file.write('Cluster '+str(i)+' - \n')
    l = clusters[i-1]

    for item in l:
        file.write(item+', ')

    file.write('\n\n')

file.write('Singletons - \n')
for item in singleton_cluster:
    file.write(item+', ')

file.write('\n\n')


#print('\n\n')
