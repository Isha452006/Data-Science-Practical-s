# Practical 4: Exploratory Data Analysis (EDA)
# Objective: Perform descriptive analysis on Iris dataset (CSV version)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# STEP 1: Load CSV Dataset
# ---------------------------
# Make sure iris.csv is in the same folder as this script
data = pd.read_csv('iris.csv')
print("✅ Dataset Loaded Successfully!\n")
print(data.head())

# ---------------------------
# STEP 2: Dataset Info & Summary
# ---------------------------
print("\n📊 Dataset Info:")
print(data.info())

print("\n📈 Statistical Summary:\n", data.describe())

# ---------------------------
# STEP 3: Check for Missing Values
# ---------------------------
print("\n🔍 Missing Values:\n", data.isnull().sum())

# ---------------------------
# STEP 4: Analyze Distributions
# ---------------------------
data.hist(figsize=(10, 6), color='skyblue', edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=16)
plt.show()

# Optional: Pairplot to visualize feature relationships
sns.pairplot(data, hue='species', palette='Set2')
plt.show()

# ---------------------------
# STEP 5: Analyze Correlations
# ---------------------------
corr_matrix = data.corr()
print("\n🔗 Correlation Matrix:\n", corr_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# ---------------------------
# STEP 6: Analyze Categorical Variables
# ---------------------------
# Value counts for species
print("\n📦 Species Value Counts:\n", data['species'].value_counts())

sns.countplot(x='species', data=data, palette='Set2')
plt.title('Count of Each Species')
plt.show()

# ---------------------------
# STEP 7: Pivot Table Example
# ---------------------------
pivot_table = pd.pivot_table(data,
                             values=['sepal length', 'sepal width', 'petal length', 'petal width'],
                             index='species',
                             aggfunc='mean')

print("\n📊 Pivot Table (Average Feature Values by Species):\n", pivot_table)
