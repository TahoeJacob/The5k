# Flow model based off cryo-rocket.com flow model.
# Data is exported directly from cearun and is in a CSV file exported from excel

# First step is to extract the date from the csv files.
# These files will be in the format of:
# O/F Temp [K] ISP 

# Need way of exporting the data from CEA from text file

# Pip install the bellow libraires

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the data from text file
testFileName = '/Users/jacobsaunders/Desktop/EngineDev/The5k/RawCSV/CEATextTest.txt'
textfile = testFileName#'/Users/jacobsaunders/Desktop/EngineDev/The5k/Archive/natetest.txt'

# Formatting text file
data = pd.read_csv(textfile, sep='\t', header=None)
# delete the first 30 rows
removeStartRows = 30
data = data.drop(data.index[0:removeStartRows])

# Create groups of data until end of file
# Split the data into columns   
pd.set_option('display.max_rows', None)
data = data[0].str.split(' ', expand=True)

# Export DataFrame to Excel file for testing purposes only
data.to_excel('output.xlsx', index=False)


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
    for startindex, endindex in zip(startindicies, endindicies):
        #print(df.iloc[startindex:endindex,:])
        sub_df = df.iloc[startindex:endindex, :]  # Extract rows from start_index to index (inclusive)
        sub_dfs[f'{len(sub_dfs)}'] = sub_df.reset_index(drop=True)
    return sub_dfs


# Define a function to convert strings with the format '2.1382-5' to floating point
def custom_float_conversion(value_str):
    try:
        if value_str is not None and '-' in value_str:
            parts = value_str.split('-')
            return float(parts[0]) * 10 ** (-int(parts[1]))
        elif value_str is not None:
            return float(value_str)
        else:
            return None
    except ValueError:
        return None

# Split the DataFrame based on starting and finishing characters
start_char = 'Pin'
end_char = 'THAN'
column_to_split = 1
sub_dfs = split_dataframe(data, start_char, end_char, column_to_split)

CEAData = {
        'Pinf/P_Chamber': [],
        'Pinf/P_Throat': [],
        'P_Chamber': [],
        'P_Throat': [],
        'T_Chamber': [],
        'T_Throat': [],
        'Rho_Chamber': [],
        'Rho_Throat': [],
        'H_Chamber': [],
        'H_Throat': [],
        'U_Chamber': [],
        'U_Throat': [],
        'G_Chamber': [],
        'G_Throat': [],
        'S_Chamber': [],
        'S_Throat': [],
        'M_Chamber': [],
        'M_Throat': [],
        '(dLV/dLP)t_Chamber': [],
        '(dLV/dLP)t_Throat': [],
        '(dLV/dlT)p_Chamber': [],
        '(dLV/dlT)p_Throat': [],
        'Cp_Chamber': [],
        'Cp_Throat': [],
        'Gamma_Chamber': [],
        'Gamma_Throat': [],
        'SonicVel_Chamber': [],
        'SonicVel_Throat': [],
        'MachNum_Chamber': [],
        'MachNum_Throat': [],
        'Viscosity_Chamber': [],
        'Viscosity_Throat': [],
        'Cp_Equil_Chamber': [],
        'Cp_Equil_Throat': [],
        'Conductivity_Equil_Chamber':[],
        'Conductivity_Equil_Throat': [],
        'Prandtl_Equil_Chamber': [],
        'Prandtl_Equil_Throat': [],
        'Cp_Froz_Chamber': [],
        'Cp_Froz_Throat': [],
        'Conductivity_Froz_Chamber': [],
        'Conductivity_Froz_Throat': [],
        'Prandtl_Froz_Chamber': [],
        'Prandtl_Froz_Throat': [],
        'O/F': [],
        'Ae/At': [],
        'Cstar': [],
        'CF': [],
        'Ivac': [],
        'Isp': [],
        }

# Key rows
# For Nate_Test data (for some reason there is an added row in 17 so everything is behind CEAData.txt)
keyRows = [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,27,28,29,31,32,33,6,35,36,37,38,39]
TwoValueRows = [8,9,10,12,13,14,15,16,17,18,19,20,21,22,25,27,28,29,31,32,33]
SingleValueRows = [6,35, 36, 37, 38, 39]

# Rows for CEATest data
# keyRows = [8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,26,28,29,30,32,33,34,6,36,37,38,39,40]
# TwoValueRows = [8,9,10,12,13,14,15,16,18,19,20,21,22,23,26,28,29,30,32,33,34]
# SingleValueRows = [6, 36, 37, 38, 39, 40]
CustomRows = [11]
keysDict = list(CEAData.keys())
print(len(keysDict))

combinedData = pd.DataFrame()
# Iterate through each sub_df and extract the data
for key, sub_df in sub_dfs.items():

    combinedData = pd.concat([combinedData, sub_df], ignore_index=False)

    # Dictionary index
    dictIndex = 0
    # Depending on each row, extract the data and append to the dictionary
    # Iterate through each row of the data
    for keyRowID in keyRows:
        firstValue = False
        skipCol = False
        # Iterate through all columns excluding first 3 
        for columnID in sub_df.columns[3:]:

            # If row has a Chamber and Exit value then check it is not empty
            if keyRowID in TwoValueRows:
                if sub_df.iloc[keyRowID, columnID] != '' and sub_df.iloc[keyRowID, columnID] != None and firstValue == False and not sub_df.iloc[keyRowID, columnID].isalpha():
                    dictKeyID = keysDict[dictIndex]
                    CEAData[dictKeyID].append(sub_df.iloc[keyRowID, columnID])
                    #print(sub_df.iloc[keyRowID, columnID], dictKeyID, keyRowID, dictIndex)
                    dictIndex +=1
                    firstValue = True
                elif sub_df.iloc[keyRowID, columnID] != '' and sub_df.iloc[keyRowID, columnID] != None and firstValue == True and not sub_df.iloc[keyRowID, columnID].isalpha():
                    dictKeyID = keysDict[dictIndex]
                    CEAData[dictKeyID].append(sub_df.iloc[keyRowID, columnID])
                    #print(sub_df.iloc[keyRowID, columnID], dictKeyID, keyRowID, dictIndex)
                    firstValue = False
                    dictIndex +=1
            elif keyRowID in CustomRows:
                if sub_df.iloc[keyRowID, columnID] != '' and sub_df.iloc[keyRowID,columnID] != None and not sub_df.iloc[keyRowID, columnID].isalpha() and firstValue == False and skipCol == False:
                    dictKeyID = keysDict[dictIndex]
                    CEAData[dictKeyID].append(sub_df.iloc[keyRowID, columnID])
                    #print(sub_df.iloc[keyRowID, columnID], dictKeyID, keyRowID, dictIndex)
                    dictIndex +=1
                    firstValue = True
                    skipCol = True
                elif skipCol == True:
                    skipCol = False
                    continue
                elif sub_df.iloc[keyRowID, columnID] != '' and sub_df.iloc[keyRowID,columnID] != None and not sub_df.iloc[keyRowID, columnID].isalpha() and sub_df.iloc[keyRowID, columnID] != '0' and firstValue == True and skipCol == False:
                    dictKeyID = keysDict[dictIndex]
                    CEAData[dictKeyID].append(sub_df.iloc[keyRowID, columnID])
                    #print(sub_df.iloc[keyRowID, columnID], dictKeyID, keyRowID, dictIndex)
                    dictIndex +=1  
                    firstValue = False 
                    break
            elif keyRowID in SingleValueRows:  
                if sub_df.iloc[keyRowID, columnID] != '' and sub_df.iloc[keyRowID, columnID] != None and not sub_df.iloc[keyRowID, columnID].isalpha():
                    dictKeyID = keysDict[dictIndex]
                    CEAData[dictKeyID].append(sub_df.iloc[keyRowID, columnID])
                    # print(sub_df.iloc[keyRowID, columnID], dictKeyID, keyRowID, dictIndex)
                    dictIndex +=1
                    break
        # After iterating through each column move onto next row and thus next dictionary index position         
    # Initialize default values for keys with '_Chamber' and '_Throat' suffixes
    key_IDs = []
    for index, row in sub_df.iloc[41:].iterrows():
        key_ID = str(sub_df.iloc[index, 1]).strip()
        print(key_ID)
        if key_ID == '':
            # Need to make sure any keys not used in this sub df but used in previous sub dfs have none added to them for this round
            keysDict = list(CEAData.keys())
            used_key_IDs = keysDict[50:]
            # Iterate through the current list of keys in the dictionary and compare against keys used in sub_df
            for key in used_key_IDs:
                if key not in key_IDs:
                    CEAData[key].append(None)
                    print(key)
            break
        # Need to track what key_IDs have been used for this specific Sub_Df
        if key_ID + '_Chamber' not in CEAData:
            CEAData[key_ID + '_Chamber'] = [None] * (len(CEAData['O/F'])-1)
            CEAData[key_ID + '_Throat'] = [None] * (len(CEAData['O/F'])-1)
            #print(CEAData[key_ID + '_Chamber'])

        molarMass_Chamber = None
        molarMass_Throat = None
        if len(key_ID) == 2:
            molarMass_Chamber = sub_df.iloc[index, 16]
            molarMass_Throat = sub_df.iloc[index, 18]
        elif len(key_ID) == 3:
            molarMass_Chamber = sub_df.iloc[index, 15]
            molarMass_Throat = sub_df.iloc[index, 17]
        elif len(key_ID) == 4:
            molarMass_Chamber = sub_df.iloc[index, 14]
            molarMass_Throat = sub_df.iloc[index, 16]
        elif len(key_ID) == 6:
            molarMass_Chamber = sub_df.iloc[index, 12]
            molarMass_Throat = sub_df.iloc[index, 14]

        CEAData[key_ID + '_Chamber'].append(molarMass_Chamber)
        CEAData[key_ID + '_Throat'].append(molarMass_Throat)

        # Append used key id with suffixes for this sub df
        key_IDs.append(key_ID+'_Chamber')
        key_IDs.append(key_ID+'_Throat')

# Create a function to display the length off all arrays
for key, value in CEAData.items():
    print(key, len(value))

combinedData.to_excel('combinedData.xlsx', index=True)
# Export the data to an Excel file
df = pd.DataFrame(CEAData)
df.to_excel('CEAParsed.xlsx', index=False)

# Apply the function to all elements of the DataFrame
df = df.applymap(custom_float_conversion)

