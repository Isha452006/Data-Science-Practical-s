# Practical 2: Numerical Computations and Broadcasting

import numpy as np

# ---------------------------
# 1. Broadcasting
# ---------------------------

# Creating arrays of different shapes
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

b = np.array([10, 20, 30])  # 1D array

# Broadcasting 'b' to each row of 'a'
broadcast_result = a + b

print("Array A:\n", a)
print("\nArray B:\n", b)
print("\nBroadcasting Result (A + B):\n", broadcast_result)

# ---------------------------
# 2. Random Arrays and Statistical Computations
# ---------------------------

# Generating random array (3x3)
rand_arr = np.random.rand(3, 3)
print("\nRandom Array:\n", rand_arr)

# Statistical operations
print("\nMean of random array:", np.mean(rand_arr))
print("Median of random array:", np.median(rand_arr))
print("Standard deviation of random array:", np.std(rand_arr))
print("Maximum value:", np.max(rand_arr))
print("Minimum value:", np.min(rand_arr))

# ---------------------------
# 3. Linear Algebra Operations
# ---------------------------

# Creating two matrices
mat1 = np.array([[2, 3], [1, 4]])
mat2 = np.array([[5, 6], [7, 8]])

# Matrix multiplication
mat_mult = np.dot(mat1, mat2)
print("\nMatrix Multiplication (mat1 x mat2):\n", mat_mult)

# Determinant
det_mat1 = np.linalg.det(mat1)
print("\nDeterminant of mat1:", det_mat1)

# Inverse
inv_mat1 = np.linalg.inv(mat1)
print("\nInverse of mat1:\n", inv_mat1)

# Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(mat1)
print("\nEigenvalues of mat1:", eigenvalues)
print("Eigenvectors of mat1:\n", eigenvectors)
