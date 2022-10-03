import pandas as pd
from langdetect import detect

df = pd.read_csv('sek-200-500.csv')

df['User Rating'] = str(df['User Rating'])

df_new = df[df['User Rating'].apply(detect).eq('en')]

df_new.to_csv('sek-400.csv', sep='\t')