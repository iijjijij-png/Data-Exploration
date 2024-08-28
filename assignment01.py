# Import necessary libraries
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Convert to a DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Display the first five rows
print("First five rows of the dataset:")
print(iris_df.head())

# Display the dataset's shape
print("\nDataset shape:")
print(iris_df.shape)

# Display summary statistics for each feature
print("\nSummary statistics for each feature:")
print(iris_df.describe())
