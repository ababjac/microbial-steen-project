import pandas as pd


df = pd.read_csv('metatranscriptomes/salazar_profiles/OM_RGC_genus_KO_profiles_metat_rarefy.tsv')
meta = pd.read_csv('metatranscriptomes/salazar_profiles/salazar_metadata.csv', encoding='ISO-8859-1')
convert = pd.read_csv('metatranscriptomes/salazar_profiles/metat_to_metag.csv')
convert_dict = dict(zip(convert['metat'].values.tolist(), convert['site'].values.tolist())) #df_locations -> meta_locations


df_locations = [c for c in df.columns.tolist() if c.__contains__('TARA')]
meta_locations = meta['site'].values.tolist()

#print(len(df_locations), len(meta_locations), len(convert))
count = 0
for loc in df_locations:
    if loc in convert_dict.keys():
        if convert_dict[loc] in meta_locations:
            count += 1
    else:
        if loc in meta_locations:
            count += 1

print(count)
