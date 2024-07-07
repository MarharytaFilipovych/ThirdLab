import random
import numpy as np
def sort_eigenvalues_eigenvectors(eigenvalues, matrix):
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    matrix = matrix[:, sort_indices]
    return eigenvalues, matrix

def get_E(eigenvalues):
    singular_values = np.sqrt(eigenvalues)
    E = np.diag(singular_values)
    return E, singular_values

def SVD(A):
    if A.shape[0] >= A.shape[1]:
        ATA = np.dot(A.T, A)
        eigenvalues, V = np.linalg.eig(ATA)
        eigenvalues, V = sort_eigenvalues_eigenvectors(eigenvalues, V)
        E, singular_values = get_E(eigenvalues)
        U = np.dot(A, V) / singular_values
    else:
        AAT = np.dot(A, A.T)
        eigenvalues, U = np.linalg.eig(AAT)
        eigenvalues, U = sort_eigenvalues_eigenvectors(eigenvalues, U)
        E, singular_values = get_E(eigenvalues)
        V = np.dot(A.T, U) / singular_values
    return U, E, V.T

def check_SVD(U, E, VT, matrix):
    reconstructed_matrix = np.dot(np.dot(U, E), VT)
    if np.allclose(matrix, reconstructed_matrix):
        print("Correct!")
    else:
        print("Incorrect!")

rows = random.randint(1, 15)
columns = random.randint(1, 15)
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

