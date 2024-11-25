# %% Importing libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

# from dataprep.datasets import load_dataset
# from dataprep.eda import create_report
# from autoviz import AutoViz_Class
# import matplotlib.pyplot as plt
# import ast
# import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error

pd.set_option('display.max_rows', 1000)
random_state = 11

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
else:
    print(f"A total of {np.sum(df['Price'] <= 0)} are less than or equal to 0.")
# %% Data Transformation

# Normalizing the column names to lowercases with underscores
df.columns = df.columns.str.lower() \
    .str.replace(r'[\(\)]', '', regex=True) \
    .str.replace(' ', '_')

# Renaming a couple of columns
df = df.rename(columns={
    'width_of_the_display': 'display_width',
    'height_of_the_display': 'display_height',
})

# Defining the feature variables and the target variable
features = [x for x in df.columns.tolist() if x != 'price']
target = 'price'

# %% Train-test split

# Target vector
y = df[target].values
# Feature matrix
X = df.drop(labels=[target], axis=1)

categorical_columns = df.select_dtypes(include=[np.object_]).columns.tolist()

# Checking feature matrix and target vector
print(f"Feature Matrix Shape: {X.shape}")
print(f"Target Vector Shape: {y.shape}")

# Step 1: Train-Test Split (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Step 2: Train-Validation Split (20% of the training data as validation set)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

print(f"Training Set Shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Validation Set Shape: X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"Test Set Shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# %% Encoding

encoder = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)],
    remainder='passthrough'  # Keep non-categorical columns as-is
)

X_train_encoded = encoder.fit_transform(X_train)
X_val_encoded = encoder.transform(X_val)
X_test_encoded = encoder.transform(X_test)

# %% Modeling

# Initialize the XGBoost regressor
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=random_state
)

# Fit the model on the training data
xgb_model.fit(X_train_encoded, y_train, eval_set=[(X_val_encoded, y_val)], verbose=True)

# Predict on validation and test sets
y_val_pred = xgb_model.predict(X_val_encoded)
y_test_pred = xgb_model.predict(X_test_encoded)

# Calculate and print RMSE for validation and test sets
rmse_val = root_mean_squared_error(y_val, y_val_pred)
rmse_test = root_mean_squared_error(y_test, y_test_pred)

print(f"Validation RMSE: {rmse_val:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")

# %% Final Model Training with Combined Train and Validation Set
import scipy
# Combine the training and validation sets
X_train_final = scipy.sparse.vstack((X_train_encoded, X_val_encoded))
y_train_final = np.hstack([y_train, y_val])

# Initialize the XGBoost regressor
xgb_model_final = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=random_state
)

# Fit the model on the combined training data (train + validation)
xgb_model_final.fit(X_train_final, y_train_final)

# Predict on the test set
y_test_pred_final = xgb_model_final.predict(X_test_encoded)

# Calculate and print RMSE for the final model on the test set
rmse_test_final = root_mean_squared_error(y_test, y_test_pred_final)

print(f"Test RMSE (Final Model): {rmse_test_final:.2f}")