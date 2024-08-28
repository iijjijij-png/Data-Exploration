# Iris Dataset Exploration

This project explores the famous Iris dataset using Python. The Iris dataset is a well-known dataset in the machine learning community, and it contains information about three different species of Iris flowers, with four features measured for each sample.

## Project Overview

The main objective of this project is to:
- Load the Iris dataset from the `sklearn.datasets` module.
- Display the first five rows of the dataset.
- Show the dataset's shape.
- Provide summary statistics (mean, standard deviation, min/max values) for each feature.

## Dataset

The Iris dataset consists of 150 samples with the following features:
- **sepal length (cm)**
- **sepal width (cm)**
- **petal length (cm)**
- **petal width (cm)**

There are three classes (species) in the dataset:
- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

## Code Explanation

### Dependencies

This project uses the following Python libraries:
- `sklearn`
- `pandas`

### Code

The code is straightforward and performs the following steps:

1. **Load the Iris dataset**:
   The dataset is loaded from the `sklearn.datasets` module using the `load_iris()` function.

2. **Convert to a DataFrame**:
   The dataset is converted to a pandas DataFrame for easier manipulation and analysis.

3. **Display the First Five Rows**:
   The first five rows of the dataset are displayed to get an initial understanding of the data.

4. **Display the Dataset's Shape**:
   The shape of the dataset (number of rows and columns) is shown.

5. **Summary Statistics**:
   The mean, standard deviation, minimum, and maximum values for each feature are calculated and displayed.

### Output

The script outputs the following:
- The first five rows of the dataset.
- The shape of the dataset.
- Summary statistics including mean, standard deviation, min, and max values for each feature.

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the dependencies using `pip`:

```bash
pip install scikit-learn pandas
