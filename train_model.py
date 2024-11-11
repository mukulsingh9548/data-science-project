import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import pickle

# Load the dataset
data = pd.read_csv('final_dataset.csv')

# Define the features
numeric_features = ['beds', 'baths', 'size']
categorical_features = ['zip_code']

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Use most frequent for zip_code since it's categorical
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

# Separate features and target
X = data[numeric_features + categorical_features]
y = data['price']  # assuming the target variable is named 'price'

# Train the model
pipe.fit(X, y)

# Save the model
with open('RidgeModel.pkl', 'wb') as f:
    pickle.dump(pipe, f)
