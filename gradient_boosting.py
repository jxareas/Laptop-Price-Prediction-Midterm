# %% Importing libraries
import pandas as pd
import numpy as np
from dataprep.datasets import load_dataset
from dataprep.eda import create_report
from autoviz import AutoViz_Class
import matplotlib.pyplot as plt
import ast
import re

pd.set_option('display.max_rows', 1000)

# %% Loading dataframe

df = pd.read_csv('data/ebay_laptops_clean.csv')

# %% Data Checks
count_df = df.shape[0]
df_no_duplicates = df.drop_duplicates()
count_df_no_duplicates = df_no_duplicates.shape[0]

print(f"Found {count_df_no_duplicates - count_df} total duplicate rows in the data frame")

# No duplicates whatsoever, so we can remove the variables above
del count_df, df_no_duplicates, count_df_no_duplicates

# Checking whether all target values are greater than zero (which is a MUST, as target variable is the laptop price)
if np.all(df['Price'] > 0):
    print("All prices found in the dataset are greater than 0.")

# %% Data Transformation

# Normalizing the column names to lowercases with underscores
df.columns = df.columns.str.lower() \
    .str.replace(r'[\(\)]', '', regex=True) \
    .str.replace(' ', '_')

df.rename(columns={
    'width_of_the_display': 'display_width',
    'height_of_the_display': 'display_height',
})

# Defining the feature variables and the target variable
features = [x for x in df.columns.tolist() if x != 'price']
target = 'price'



