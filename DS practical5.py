# Practical 5: Data Visualization using Matplotlib (All in One Figure)

import matplotlib.pyplot as plt
import numpy as np

# Data for plots
x = np.linspace(0, 10, 100)
y = np.sin(x)
categories = ['A', 'B', 'C', 'D', 'E']
values = [5, 7, 3, 8, 4]
data = np.random.randn(1000)
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)

# Create figure and subplots (2 rows, 2 columns)
plt.figure(figsize=(10, 8))

# ------------------- Line Plot -------------------
plt.subplot(2, 2, 1)
plt.plot(x, y, color='purple', linestyle='--', label='sin(x)')
plt.title('Line Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

# ------------------- Bar Plot -------------------
plt.subplot(2, 2, 2)
plt.bar(categories, values, color='skyblue', edgecolor='black')
plt.title('Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')

# ------------------- Histogram -------------------
plt.subplot(2, 2, 3)
plt.hist(data, bins=30, color='orange', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# ------------------- Scatter Plot -------------------
plt.subplot(2, 2, 4)
plt.scatter(x_scatter, y_scatter, color='green')
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')

# Overall title for the figure
plt.suptitle('Data Visualization using Matplotlib', fontsize=16, fontweight='bold')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Display all plots together
plt.show()
