import py

from data_preprocessing import preprocess_data
from model_building import grid_search, train_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
txsn_dataset = pd.read_csv("Fraud.csv")

# Preprocess the data
cleaned_dataset = preprocess_data(txsn_dataset)

# Separate the dependent and independent variables
indep_var = cleaned_dataset.iloc[:, :-2].values
dep_var = cleaned_dataset.iloc[:, -2].values

# One-Hot Encoding
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1, 3, 6])],
    remainder="passthrough"
)
indep_var_encoded = ct.fit_transform(indep_var)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(indep_var_encoded, dep_var, test_size=0.2, random_state=42)

# Perform Grid Search and obtain best model
best_model = grid_search(x_train, y_train)

# Train the final model
best_model_params = best_model.get_params()
final_model = train_model(x_train, y_train, best_model_params)

# Make predictions using the final model
y_pred = final_model.predict(x_test)

# Display predicted labels
print("Predicted labels for the test set:", y_pred)
