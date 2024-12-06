import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import lil_matrix
import scipy.linalg
import os
import time
import math

def generate_mesh(domain):

    n_cells_x = domain["n_cells_x"]
    n_cells_y = domain["n_cells_y"]
    n_vertices_x = n_cells_x + 1
    n_vertices_y = n_cells_y + 1
    n_cells = n_cells_x * n_cells_y

    x_min = domain["x_min"]
    x_max = domain["x_max"]
    y_min = domain["y_min"]
    y_max = domain["y_max"]

    x, y = np.meshgrid(np.linspace(x_min, x_max, n_vertices_x), np.linspace(y_min, y_max, n_vertices_y))
    vertices = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    cells = np.zeros([n_cells, 4], dtype=np.int64)
    for j in range(0, n_cells_y):
        for i in range(0, n_cells_x):
            k = i + n_cells_x*j  

            cells[k, 0] = (i) + (n_cells_x + 1)*(j)  # the linear index of the lower left corner of the element
            cells[k, 1] = (i+1) + (n_cells_x + 1)*(j)  # the linear index of the lower right corner of the element
            cells[k, 2] = (i) + (n_cells_x + 1)*(j+1)  # the linear index of the upper right corner of the element
            cells[k, 3] = (i+1) + (n_cells_x + 1)*(j+1)  # the linear index of the upper right corner of the element

    delta = (x_max - x_min) / n_cells_x
    return vertices, cells, delta

def compute_local_mass_matrix():
    """Int_omega BiBk dx = Int_Omega BjBl dy; Multiply by h"""
    M_local = np.zeros([2, 2])
    M_local[0, 0] = 0.5
    M_local[1, 1] = 0.5
    
    return M_local

def compute_local_stiffness_matrix():
    """Int_omega Bix Bkx dx = Int_Omega Bjy Bly dy; Multiply by 1/h"""
    N_local = np.ones([2, 2])
    N_local[0, 1] = -1.0
    N_local[1, 0] = -1.0
    
    return N_local

def compute_local_advection_matrix():
    """Int_omega Bix Bk dx = Int_Omega Bjy Bl dy; Multiply by 1"""
    A_local = 0.5*np.ones([2, 2])
    A_local[0, 0] = -0.5
    A_local[1, 0] = -0.5
    
    return A_local

def compute_A1_and_A2_local():
    A1_A2 = np.zeros([4, 4])
    S_local_1D = compute_local_stiffness_matrix()
    M_local_1D = compute_local_mass_matrix()
    A1 = np.kron(S_local_1D, M_local_1D)
    A2 = np.kron(M_local_1D, S_local_1D)
    return A1, A2

def compute_global_A_1_A_2(vertices, cells):

    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]

    N_row_idx = np.zeros([n_cells, 4, 4])
    N_col_idx = np.zeros([n_cells, 4, 4]) 
    N_data = np.zeros([n_cells, 4, 4])

    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()


    A1, A2 = compute_A1_and_A2_local()

    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = np.meshgrid(cell, cell)
        N_row_idx[cell_idx, :, :] = row_idx
        N_col_idx[cell_idx, :, :] = col_idx
        N_data[cell_idx, :, :] = A1 + A2 # * delta_x[cell_idx] / delta_y[cell_idx] + A2 * delta_y[cell_idx] / delta_x[cell_idx]
        
    A_global = scipy.sparse.csr_array((N_data.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))
    
    return A_global

def set_boundary(n_cells_x, n_cells_y, F, matrices, phi_left, phi_right, phi_bottom, phi_top):

    # Left boundary
    i_idx = np.zeros(n_cells_y + 1, dtype=np.int64)
    j_idx = np.arange(0, n_cells_y + 1)
    matrices = handle_boundary(i_idx, j_idx, n_cells_x, F, matrices, phi_left)

    # Right boundary
    i_idx = n_cells_x*np.ones(n_cells_y + 1, dtype=np.int64)
    j_idx = np.arange(0, n_cells_y + 1)
    matrices = handle_boundary(i_idx, j_idx, n_cells_x, F, matrices, phi_right)

    # Bottom boundary
    i_idx = np.arange(0, n_cells_x + 1)
    j_idx = np.zeros(n_cells_x + 1, dtype=np.int64)
    matrices = handle_boundary(i_idx, j_idx, n_cells_x, F, matrices, phi_bottom)

    # Top boundary
    i_idx = np.arange(0, n_cells_x + 1)
    j_idx = n_cells_y*np.ones(n_cells_x + 1, dtype=np.int64)
    matrices = handle_boundary(i_idx, j_idx, n_cells_x, F, matrices, phi_top)

def handle_boundary(i_idx, j_idx, n_cells_x, F, matrices, phi):
        
        basis_indices = i_idx + (n_cells_x + 1)*j_idx
        for matrix in matrices:
            matrix[basis_indices, :] = 0.0
            matrix[basis_indices, basis_indices] = 1.0
        
        F[basis_indices]  = phi
        return matrices

def compute_boundary_indices(n_cells_x, n_cells_y):
    # Initialize basis_indices as an empty list
    basis_indices = []
    
    # Left boundary
    i_idx = np.zeros(n_cells_y + 1, dtype=np.int64)
    j_idx = np.arange(0, n_cells_y + 1)
    basis_indices.append(i_idx + (n_cells_x + 1)*j_idx)
    
    # Right boundary
    i_idx = n_cells_x*np.ones(n_cells_y + 1, dtype=np.int64)
    j_idx = np.arange(0, n_cells_y + 1)
    basis_indices.append(i_idx + (n_cells_x + 1)*j_idx)
    
    # Bottom boundary
    i_idx = np.arange(0, n_cells_x + 1)
    j_idx = np.zeros(n_cells_x + 1, dtype=np.int64)
    basis_indices.append(i_idx + (n_cells_x + 1)*j_idx)
    
    # Top boundary
    i_idx = np.arange(0, n_cells_x + 1)
    j_idx = n_cells_y*np.ones(n_cells_x + 1, dtype=np.int64)
    basis_indices.append(i_idx + (n_cells_x + 1)*j_idx)
    
    # Convert basis_indices to a numpy array
    basis_indices = np.concatenate(basis_indices)
    	
    return basis_indices

def set_boundary_with_elimination(n_cells_x, n_cells_y, matrix, F):

    basis_indices = compute_boundary_indices(n_cells_x, n_cells_y)
    	
    A_reduced, F_reduced = eliminate_dirichlet_boundary(matrix, F, basis_indices)
    return A_reduced, F_reduced

def eliminate_dirichlet_boundary(A, b, boundary_indices):
    """
    Eliminate the rows and columns corresponding to Dirichlet boundary conditions.
    
    Parameters:
    A - The system matrix (sparse matrix)
    b - The right-hand side vector
    boundary_indices - The indices where Dirichlet conditions are applied
    
    Returns:
    A_reduced - The reduced system matrix
    b_reduced - The reduced right-hand side vector
    """

    mask = np.ones(A.shape[0], dtype=bool)
    mask[boundary_indices] = False

    A_reduced = A[mask][:, mask]
    b_reduced = b[mask]

    return A_reduced, b_reduced

def compute_Forcing(vertices, cells):
    f = lambda x: -1.0

    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]
    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()
    
    F = np.zeros(n_vertices)
    for cell_idx, cell in enumerate(cells):
        f_at_cell_vertices = f(vertices[cell])
        F[cell] += 0.25 * f_at_cell_vertices * delta_x[cell_idx]* delta_y[cell_idx]
    return F

def update_boundary_S2(vertices_boundary_S2_in_S1, n_cells_x_S2, n_cells_y, u_S1, F2, vertices_S1):
    i_idx = np.zeros(n_cells_y + 1, dtype=np.int64)
    j_idx = np.arange(0, n_cells_y + 1)
    basis_indices = i_idx + (n_cells_x_S2 + 1)*j_idx
    for i in range(0, len(vertices_boundary_S2_in_S1)):
        F2[basis_indices[i]] = u_S1[vertices_boundary_S2_in_S1[i]]
        F2[basis_indices[i] + int(len(F2)/2)] = u_S1[vertices_boundary_S2_in_S1[i] + vertices_S1.shape[0]]
    return F2

def update_boundary_S1(vertices_boundary_S1_in_S2, n_cells_x_S1, n_cells_y, u_S2, F1, vertices_S2):
    # Right boundary
    i_idx = n_cells_x_S1*np.ones(n_cells_y + 1, dtype=np.int64)
    j_idx = np.arange(0, n_cells_y + 1)
    basis_indices = i_idx + (n_cells_x_S1 + 1)*j_idx
    for i in range(0, len(vertices_boundary_S1_in_S2)):
        F1[basis_indices[i]] = u_S2[vertices_boundary_S1_in_S2[i]]
        F1[basis_indices[i] + int(len(F1)/2)] = u_S2[vertices_boundary_S1_in_S2[i] + vertices_S2.shape[0]]
    return F1

def is_symmetric(A, tol=1e-8):
    return np.allclose(A, A.T, atol=tol)

def is_positive_definite(A):
    if not is_symmetric(A):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
    
def is_symmetric_sparse(A, tol=1e-8):
    return (A - A.transpose()).nnz == 0

def is_positive_definite_sparse(A):
    if not is_symmetric_sparse(A):
        return False
    
    try:
        # Try to compute the Cholesky decomposition
        _ = scipy.linalg.cholesky(A.toarray())
        return True
    except np.linalg.LinAlgError:
        return False

def plot_velocity(u, domain, filename):
    inner_vertices = np.array(domain["inner_vertices"])
    n_cells_x = domain["n_cells_x"]
    n_cells_y = domain["n_cells_y"]

    X = inner_vertices[:, 0].reshape(n_cells_y-1, n_cells_x-1)
    Y = inner_vertices[:, 1].reshape(n_cells_y-1, n_cells_x-1)
    U1 = u.reshape(n_cells_y-1, n_cells_x-1)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X, Y, U1)
    plt.colorbar(label="U")
    plt.title(f"U1/2 Field - Cells: {n_cells_x}x{n_cells_y}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"figures/{filename}")
    
def construct_restriction_operator_old(domain, local_vertices): #takes to long
    # Extract inner vertices of the domain
    global_vertices = domain["inner_vertices"]

    # Get the number of local and global vertices
    num_local = len(local_vertices)
    num_global = len(global_vertices)

    # Initialize a sparse matrix for the restriction operator in LIL format
    R = lil_matrix((num_local, num_global))

    # Loop over each local vertex
    for local_idx, local_vertex in enumerate(local_vertices):
        # Loop over each global vertex
        for global_idx, global_vertex in enumerate(global_vertices):
            # Check if the local vertex is close to the global vertex within a tolerance
            if np.allclose(local_vertex, global_vertex, atol=1e-5):
                # Set the corresponding entry in the restriction matrix to 1
                R[local_idx, global_idx] = 1
                
    # Create a mask for rows with non-zero entries
    non_zero_row_mask = R.getnnz(axis=1) > 0

    # Filter out the rows without non-zero entries (needed because the subdomains are generated before elimination of the bc)
    R = R[non_zero_row_mask]
    
    # Convert the matrix to Compressed Sparse Row format for efficient arithmetic operations
    return R.tocsr()


