import pandas as pd
import numpy as np

arr = pd.read_csv('filtered-data/sek-900-1000  FILTERED.csv', header=None, dtype=object).values

for i in range (1,len(arr)):
    arr[i, 9] = str(round(float(arr[i, 9]) * 0.093, 2))

print(arr)

pd.DataFrame(arr).to_csv('filtered-data/eur-90-100.csv', header=None, index=None)

