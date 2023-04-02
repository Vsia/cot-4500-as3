#Vallesia Pierre Louis
#Assignment 3
import numpy as np


#Number 1- Euler Method with the following details
def euler_method(f, t_range, y0, num_iterations):
  h = (t_range[1] - t_range[0]) / num_iterations
  t = t_range[0]
  y = y0
  for i in range(num_iterations):
    y = y + h * f(t, y)
    t = t + h
  return y

def f(t, y):
  return t - y**2

t_range = [0, 2]
y0 = 1
num_iterations = 10

result = euler_method(f, t_range, y0, num_iterations)
print("{:.5f}".format(result))
print("\n")
#---------------------------------------------------------------------------------------------------
#Number 2 - Runge-Kutta with the following details:
def rungeKutta(f, t_range, y0, num_iterations):
  h = (t_range[1] - t_range[0]) / num_iterations
  t = t_range[0]
  y = y0
  for i in range(num_iterations):
    k1 = h * f(t, y)
    k2 = h * f(t + h / 2, y + k1 / 2)
    k3 = h * f(t + h / 2, y + k2 / 2)
    k4 = h * f(t + h, y + k3)
    y = y + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t = t + h
  return y

def f(t, y):
  return t - y**2

t_range = [0, 2]
y0 = 1
num_iterations = 10

result = rungeKutta(f, t_range, y0, num_iterations)
print("{:.5f}".format(result))
print("\n")
#---------------------------------------------------------------------------------------------------
#number 3- Use Gaussian elimination and backward substitution solve the following linear system of equations written in augmented matrix format.

# Define the augmented matrix for the system of equations
# Define the matrix and vector
A = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]])
b = np.array([6, 0, -3])

# Combine them into an augmented matrix
augmented_matrix = np.column_stack((A, b))

# Perform Gaussian elimination
for i in range(len(augmented_matrix)):
  # Find the row with the largest pivot element
  pivot_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i
  # Swap the current row with the pivot row
  augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]
  # Eliminate the lower triangular entries
  for j in range(i + 1, len(augmented_matrix)):
    factor = augmented_matrix[j][i] / augmented_matrix[i][i]
    augmented_matrix[j] = augmented_matrix[j] - factor * augmented_matrix[i]
# Perform backward substitution
x = np.zeros(len(b))
for i in range(len(b) - 1, -1, -1):
  x[i] = augmented_matrix[i][-1]
  for j in range(i + 1, len(b)):
    x[i] -= augmented_matrix[i][j] * x[j]
  x[i] /= augmented_matrix[i][i]

print(x, "\n")

#---------------------------------------------------------------------------------------------------
#number 4 - Implement LU Factorization for the following matrix and do the following:
#Print out the matrix determinant

def luFactorization(matrixFour):
  n = matrixFour.shape[0]
  L = np.eye(n)
  U = np.zeros((n, n))
  for j in range(n):
    U[j, j:] = matrixFour[j, j:] - L[j, :j] @ U[:j, j:]
    L[j + 1:, j] = (matrixFour[j + 1:, j] - L[j + 1:, :j] @ U[:j, j]) / U[j, j]
  return L, U


matrixFour = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2],
                       [-1, 2, 3, -1]])

L, U = luFactorization(matrixFour)
determinatnA = np.prod(np.diag(U))

print("{:.5f}".format(determinatnA))
print("\n", L, "\n")
print(U, "\n")

#---------------------------------------------------------------------------------------------------
#number 5- Determine if the following matrix is diagonally dominate.
matrixFive = np.array([[9, 3, 0, 4, 3], [0, 9, 1, 2, 2], [2, 2, 2, 12, 0],
                       [1, 1, 3, 2, 8]])


def diagonallyMatrix(matrixFive):
  for i, row in enumerate(matrixFive):
    s = sum(abs(v) for j, v in enumerate(row) if i != j)
    if s > abs(row[i]):
      return False
    return True


if (diagonallyMatrix(matrixFive)):
  print("True")
  print("\n")
else:
  print("False")
  print("\n")

#---------------------------------------------------------------------------------------------------
#Number 6- Determine if the matrix is a positive definite.

#create maxtrix
matrixSix = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
#check eigen values
eigenValue = np.all(np.linalg.eigvals(matrixSix) > 0)

if (eigenValue):
  print(eigenValue)
else:
  print(eigenValue)
