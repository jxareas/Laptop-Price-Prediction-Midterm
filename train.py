# %% Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

random_state = 11

data_path = 'data/ebay_laptops_dataset.csv'
df = pd.read_csv(data_path)
df.head()

# Renaming the dataframe columns
df.columns = df.columns.str.lower() \
    .str.replace(r'[\(\)]', '', regex=True) \
    .str.replace(' ', '_')
# %% Preprocessing
# Renaming a couple of columns
df = df.rename(columns={
    'width_of_the_display': 'display_width',
    'height_of_the_display': 'display_height',
})  # Renaming the dataframe columns
df.columns = df.columns.str.lower() \
    .str.replace(r'[\(\)]', '', regex=True) \
    .str.replace(' ', '_')

# Renaming a couple of columns
df = df.rename(columns={
    'width_of_the_display': 'display_width',
    'height_of_the_display': 'display_height',
})

# Defining the feature variables and the target variable
features = ['brand', 'color', 'condition', 'gpu', 'processor',
            'processor_speed', 'processor_speed_unit', 'type', 'display_width',
            'display_height', 'os', 'storage_type', 'hard_drive_capacity',
            'hard_drive_capacity_unit', 'ssd_capacity', 'ssd_capacity_unit',
            'screen_size_inch', 'ram_size', 'ram_size_unit']
target = 'price'

# %% Train-Test-Validation Split


# Columns to be selected, which are comprised of the features and the target variable
columns = features + [target]

# First split: 80% for training + validation, 20% for testing
df_full_train, df_test = train_test_split(df[columns], test_size=0.2, random_state=random_state)

# Second split: Split the training data into 75% train and 25% validation (which results in 60% train and 20% validation overall)
df_train, df_val = train_test_split(df_full_train[columns], test_size=0.25, random_state=random_state)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train[target]
y_train = df_train[target]
y_val = df_val[target]
y_test = df_test[target]

# Removing the target variable from each of the dataframes used to perform model training / evaluation
del df_train[target], df_val[target], df_test[target], df_full_train[target]


# %% Model Training
def evaluate_regression_model(title, model, param_grid, X_train, y_train, X_val, y_val, cv=5,
                              scoring='neg_mean_squared_error'):
    """
    Evaluates a linear regression model by performing hyperparameter tuning using GridSearchCV,
    training the model, and reporting the evaluation metrics (RMSE, MAE, and R²) on the validation set.

    Parameters:
    title (str): The title or name of the model, used for printing and logging.
    model (sklearn estimator): The regression model to be evaluated.
    param_grid (dict): A dictionary containing the hyperparameters to be tuned.
    X_train (array-like or pandas DataFrame): The training features.
    y_train (array-like or pandas Series): The training target variable.
    X_val (array-like or pandas DataFrame): The validation features.
    y_val (array-like or pandas Series): The validation target variable.
    cv (int, optional): The number of folds in cross-validation. Default is 5.
    scoring (str, optional): The scoring metric for GridSearchCV. Default is 'neg_mean_squared_error'.

    Returns:
    tuple: A tuple containing the best hyperparameters and the best model found by GridSearchCV.

    Raises:
    Exception: If an error occurs during the evaluation, an error message is printed.

    Example:
    >>> model = Ridge()
    >>> param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    >>> best_params, best_model = evaluation_model_regression("Ridge", model, param_grid, X_train, y_train, X_val, y_val)
    """
    try:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        print(f"{title} - Best Hyperparameters: {best_params}")

        y_pred = best_model.predict(X_val)

        # Evaluation metrics
        rmse = root_mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"{title} Evaluation Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")

        return best_params, best_model

    except Exception as e:
        print(f"An error occurred while evaluating the model: {str(e)}")


categorical_columns = df_train.select_dtypes(include=[np.object_]).columns.tolist()
numerical_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()

# Define a pipeline for numerical data (Impute missing values with median)
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# Define a pipeline for categorical data (One-hot encode the categorical columns)
categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both pipelines into a single column transformer
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_columns),
    ('categorical', categorical_pipeline, categorical_columns)
])

# Apply the transformations to the training and validation data
X_train = preprocessor.fit_transform(df_train)
X_val = preprocessor.transform(df_val)
X_test = preprocessor.transform(df_test)

# Parameter grid for RandomForestRegressor
param_grid_rf = {
    'n_estimators': [10, 15],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Train and evaluate RandomForestRegressor
best_params_rf, best_model_rf = evaluate_regression_model(
    "Random Forest Regressor",
    RandomForestRegressor(random_state=random_state),
    param_grid_rf,
    X_train,
    y_train,
    X_val,
    y_val
)

# %% Final Model Training
X_full_train = preprocessor.transform(df_full_train)

best_model_rf = RandomForestRegressor(
    n_estimators=best_params_rf['n_estimators'],
    max_depth=best_params_rf['max_depth'],
    min_samples_split=best_params_rf['min_samples_split'],
    min_samples_leaf=best_params_rf['min_samples_leaf'],
    random_state=random_state
)
best_model_rf.fit(X_full_train, y_full_train)

# In[19]:


# Predict on the test set
y_test_pred_rf = best_model_rf.predict(X_test)

# Calculate evaluation metrics
rmse_rf = root_mean_squared_error(y_test, y_test_pred_rf)  # RMSE
mae_rf = mean_absolute_error(y_test, y_test_pred_rf)  # MAE
r2_rf = r2_score(y_test, y_test_pred_rf)  # R²

# Print the evaluation metrics
print("Random Forest Regressor Evaluation on Test Set:")
print(f"  RMSE: {rmse_rf:.4f}")
print(f"  MAE: {mae_rf:.4f}")
print(f"  R²: {r2_rf:.4f}")


# %% Saving the model

def save_model(model, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump((model, preprocessor), f_out)
    return output_file


model_filename = 'laptop_price_rf.bin'
input_file = save_model(best_model_rf, model_filename)

print(f'Model and preprocessor successfully saved to {model_filename}')
