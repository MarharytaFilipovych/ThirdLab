import random
import numpy as np
def SVD(matrix):
    A = np.array(matrix)
    AAT = np.dot(A, A.T)
    eigenvalues, U = np.linalg.eig(AAT)
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    U = U[:, sort_indices]
    singular_values = np.sqrt(eigenvalues)
    E = np.diag(singular_values)
    V = np.dot(A.T, U) / singular_values
    return U, E, V.T

def check_SVD(U, E, VT, matrix):
    reconstructed_matrix = np.dot(np.dot(U, E), VT)
    if np.allclose(matrix, reconstructed_matrix):
        print("Correct!")
        return True
    print("Incorrect!")
    return False

#matrix = np.array([[5, 1], [8, 1]])
rows = columns = random.randint(2, 4)
matrix = np.random.randint(0, 256, size=(rows, columns))
U, E, VT = SVD(matrix)
print("Original Matrix:")
print(matrix)
print(f"U:")
print(U)
print(f"E:")
print(E)
print(f"VT:")
print(VT)

check_SVD(U, E, VT, matrix)

