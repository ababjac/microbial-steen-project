import pandas as pd

meta = pd.read_csv('metatranscriptomes/salazar_profiles/salazar_metadata.csv', encoding='ISO-8859-1')
print(meta['Depth.nominal'])
