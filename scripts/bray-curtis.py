import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cut_tree, fcluster
import matplotlib.pyplot as plt

df = pd.read_csv('metatranscriptomes/salazar_profiles/OM_RGC_genus_KO_profiles_metat_rarefy.tsv')

#normalize by totals from columns
norm_df = pd.DataFrame()

for c in df.columns:
    if c.__contains__('TARA'):
        total = df.loc[:, c].sum()

        if total == 0:
            norm_df[c] = df[c] #all values are 0 cannot divide
            continue

        norm_df[c] = df[c] / total

#print(norm_df)

#calculate upper triangular distance matrix
matrix = pdist(norm_df, 'braycurtis')
print(matrix)

#calculate linkages
links = linkage(matrix, 'single')
print(links)


######################################################################
#PREVIOUS CODE
######################################################################

# print('Picking threshold with CH_index')

# if FILENAME.__contains__('HD'):
#     thresholds = np.arange(0.005, 0.1, 0.005)
# elif FILENAME.__contains__('KS'):
#     thresholds = np.arange(0.1, 0.5, 0.01)

#thresholds = np.arange(0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000001, 0.00001, 0.01)
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
#
# print('Plotting Elbow...')
# #Plot the elbow
# plt.plot(thresholds, errors, 'bx-')
# plt.xlabel('Threshold Value')
# plt.ylabel('CH Index')
# plt.title('Optimal Threshold Elbow Plot')
# plt.savefig(ERRORPLOT)
# plt.close()
#
# print('Plotting Dendrogram...')
# colors = cm.gist_ncar(np.arange(0, 1, 0.05))
#
# colorlst=[]# empty list where you will put your colors
# for i in range(len(colors)): #get for your color hex instead of rgb
#     colorlst.append(col.to_hex(colors[i]))
#
# set_link_color_palette(colorlst)
#
# D = dendrogram(Z,labels=genes, color_threshold=THRESHOLD, above_threshold_color='gray')
# plt.axhline(y=THRESHOLD, c='gray', lw=1, linestyle='dashed')
#
# if COLNAME == 'Gene':
#     plt.title('E. Coli')
# else:
#     plt.title('Yeast')
#
# plt.savefig(DENDROGRAMPLOT)
# plt.close()

print('Creating clusters using fcluster')
clusters = fcluster(Z, THRESHOLD, criterion='distance')

data = {'Gene' : genes, 'cluster_id' : list(clusters)}
df = pd.DataFrame(data)
length = len(set(clusters))

OUT = OUTFILE+'_t'+str(round(THRESHOLD, 3))+'.txt'

clusters = []
singleton_cluster = []
for i in range(1, length+1):
    l = df['Gene'][df['cluster_id'] == i].values.tolist()

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


print('\n\n')
