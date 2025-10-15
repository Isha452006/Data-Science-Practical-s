# Practical 6: Advanced Data Visualization using Seaborn (Soft Light Version)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = sns.load_dataset('iris')

# Set a light theme with soft gridlines
sns.set_theme(style="whitegrid")

# ------------------------------
# 1️⃣ Correlation Heatmap (very light colors)
# ------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='BuGn', linewidths=0.5)
plt.title('Correlation Heatmap of Iris Dataset', fontsize=12)
plt.show()

# ------------------------------
# 2️⃣ Pairplot (soft pastel colors)
# ------------------------------
sns.pairplot(df, hue='species', palette='pastel', corner=True)
plt.suptitle('Pairplot of Iris Dataset', y=1.02, fontsize=12)
plt.show()

# ------------------------------
# 3️⃣ Boxplot (light pastel tones)
# ------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x='species', y='sepal_length', data=df, palette='light:#AEDFF7')
plt.title('Boxplot of Sepal Length by Species', fontsize=12)
plt.show()

# ------------------------------
# 4️⃣ Combine Multiple Seaborn Plots (All Light)
# ------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Heatmap (light teal)
sns.heatmap(df.corr(numeric_only=True), ax=axes[0], cmap='BuGn', cbar=False, linewidths=0.5)
axes[0].set_title('Heatmap', fontsize=11)

# Boxplot (light blue)
sns.boxplot(x='species', y='petal_length', data=df, ax=axes[1], palette='light:#C7E8FF')
axes[1].set_title('Boxplot', fontsize=11)

# Scatterplot (light pastel dots)
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df, ax=axes[2],
                palette=['#A9E5BB', '#F7D6BF', '#C7E8FF'], edgecolor='none')
axes[2].set_title('Scatterplot', fontsize=11)

plt.suptitle('Combined Seaborn Plots (Soft Light Theme)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
