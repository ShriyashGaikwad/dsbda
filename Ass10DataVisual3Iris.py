## Data Visualization III

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# iris = load_iris()
# data = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
# data['species'] = iris.target

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, names=columns)

## Display dataset details (info and first few rows)

print("Dataset Information:")
print(data.info())
print("\nFirst 5 rows of the dataset:")
print(data.head())

## Calculate Min, Max, Mean, Variance value and Percentiles (Quantiles)

# Drop the categorical 'species' column to calculate numerical statistics
numeric_data = data.drop('species', axis=1)

print("--- Minimum Values ---")
print(numeric_data.min(), "\n")

print("--- Maximum Values ---")
print(numeric_data.max(), "\n")

print("--- Mean Values ---")
print(numeric_data.mean(), "\n")

print("--- Variance Values ---")
print(numeric_data.var(), "\n")

print("--- Percentiles using Quantile ---")
print("25th Percentile (0.25):")
print(numeric_data.quantile(0.25), "\n")
print("50th Percentile (Median/0.50):")
print(numeric_data.quantile(0.50), "\n")
print("75th Percentile (0.75):")
print(numeric_data.quantile(0.75))

## Display the Histogram using Hist Function
numeric_data.hist(figsize=(10, 8))
plt.show()

## Display the Boxplot using Boxplot Function

# Display the boxplot using the boxplot() function to identify outliers
data.boxplot(column=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.show()