# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:03:57 2024


This script is used for concat all results
PRESS F5 TO RUN ALL THE CODE OR RUN VIA CMD USING --> python summary_results.py
"""
print('#' * 50)
print('Summary Results'.center(50))
print('#' * 50)


# Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
import glob
print('Libraries imported')
# Check for the path where is results
folder = Path(__file__)
xlsx_files = list(folder.parents[0].glob('*.xlsx*'))
# Load all folders
files = {}
for i in xlsx_files:
    files[i] = pd.read_excel(i)
    print('File loaded -->' , i)
# Concat all files
df = pd.DataFrame()
for i in files.keys():
    df = pd.concat([df , files[i]] , axis = 0)
print(df.columns)
# Pivots by model and set
td = pd.pivot_table(df ,
                    values = ['RMSE' , 'MAE'],
                    index = ['Modelo' , 'Tipo_Resultado'],
                    aggfunc = 'mean').reset_index()
# Export
with pd.ExcelWriter(folder.parents[0] / 'summary_results.xlsx') as export:
    df.to_excel(export , sheet_name = 'Detail')
    td.to_excel(export , sheet_name = 'Pivot')
print('File generated -->' , folder.parents[0] / 'summary_results.xlsx')
print('Process finished c:')