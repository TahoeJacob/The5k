# Flow model based off cryo-rocket.com flow model.
# Data is exported directly from cearun and is in a CSV file exported from excel

# First step is to extract the date from the csv files.
# These files will be in the format of:
# O/F Temp [K] ISP 

# Need way of exporting the data from CEA from text file

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import math

# Importing the data from text file
textfile = '/Users/jacobsaunders/Desktop/EngineDev/The5k/natetest.txt'
data = pd.read_csv(textfile, sep='\t', header=None)
# delete the first 20 rows
removeStartRows = 30
data = data.drop(data.index[0:removeStartRows])

# Create groups of data until end of file
# Split the data into columns   
pd.set_option('display.max_rows', None)
data = data[0].str.split(' ', expand=True)

# Function to split DataFrame based on starting and finishing characters in Column1
def split_dataframe(df, start_char, end_char, column_to_split):
    sub_dfs = {}
    indices = []
    startindicies = []
    endindicies = []

    for index, row in df.iterrows():
        
        if row[column_to_split].startswith(start_char) and row[2].startswith('='):
            startindicies.append(index-removeStartRows)
        if row[column_to_split].endswith(end_char) and row[2].endswith('1.'):
            endindicies.append(index-removeStartRows)
    print(startindicies, endindicies)
    for startindex, endindex in zip(startindicies, endindicies):
        #print(df.iloc[startindex:endindex,:])
        sub_df = df.iloc[startindex:endindex, :]  # Extract rows from start_index to index (inclusive)
        sub_dfs[f'{len(sub_dfs)}'] = sub_df.reset_index(drop=True)
    return sub_dfs

# Split the DataFrame based on starting and finishing characters
start_char = 'Pin'
end_char = 'THAN'
column_to_split = 1
sub_dfs = split_dataframe(data, start_char, end_char, column_to_split)



for key, sub_df in sub_dfs.items():
    # extract the data from the sub_df into a new dataframe




