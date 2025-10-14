# Practical 1: NumPy Basics and Array Operations

import numpy as np

# ---------------------------
# 1. Creating Arrays
# ---------------------------

# 1D array
arr1 = np.array([10, 20, 30, 40, 50])
print("1D Array:\n", arr1)

# 2D array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:\n", arr2)

# ---------------------------
# 2. Indexing and Slicing
# ---------------------------

# Indexing in 1D array
print("\nFirst element of arr1:", arr1[0])
print("Last element of arr1:", arr1[-1])

# Slicing in 1D array
print("Elements from index 1 to 3:", arr1[1:4])

# Indexing in 2D array
print("\nElement at row 1, column 2:", arr2[0, 1])

# Slicing in 2D array
print("First two rows and first two columns:\n", arr2[:2, :2])

# ---------------------------
# 3. Element-wise Operations
# ---------------------------

arr3 = np.array([5, 10, 15, 20, 25])
print("\nElement-wise addition:", arr1 + arr3)
print("Element-wise subtraction:", arr1 - arr3)
print("Element-wise multiplication:", arr1 * arr3)
print("Element-wise division:", arr1 / arr3)

# ---------------------------
# 4. Matrix Operations
# ---------------------------

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

print("\nMatrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)

# Matrix addition
print("Matrix Addition:\n", matrix1 + matrix2)

# Matrix multiplication (dot product)
print("Matrix Multiplication (dot):\n", np.dot(matrix1, matrix2))

# ---------------------------
# 5. Statistical Functions
# ---------------------------

print("\nMean of arr1:", np.mean(arr1))
print("Standard Deviation of arr1:", np.std(arr1))
print("Dot Product of arr1 and arr3:", np.dot(arr1, arr3))
