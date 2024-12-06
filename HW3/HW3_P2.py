import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.linalg
import os
import time

def generate_mesh(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y):
    
    n_vertices_x = n_cells_x + 1
    n_vertices_y = n_cells_y + 1
    n_cells = n_cells_x * n_cells_y

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

    return vertices, cells

def compute_local_mass_matrix():
    """Int_omega BiBk dx = Int_Omega BjBl dy; Multiply by h"""
    M_local = np.zeros([2, 2])
    M_local[0, 0] = 0.5
    M_local[1, 1] = 0.5
    
    return M_local

def compute_local_stiffness_matrix():
    """Int_omega Bix Bkx dx = Int_Omega Bjy Bly dy; Multiply by 1/dx * 1/dy"""
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
        N_data[cell_idx, :, :] = A1 + A2
    
    A_global = scipy.sparse.csr_array((N_data.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))
    
    return A_global

def compute_Forcing(vertices, delta_x, delta_y):
    return 0.25 * np.ones(len(vertices)) * delta_x* delta_y

def set_boundary_conditions(n_cells_x, n_cells_y, F, matrices, phi_left, phi_right, phi_bottom, phi_top):

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
        
        F[basis_indices] = F[basis_indices+int(len(F)/2)] = phi
        return matrices

def handle_Schwartz_boundary(i_idx, j_idx, n_cells_x, F, matrices, phi):
        basis_indices = i_idx + (n_cells_x + 1)*j_idx
        
        for matrix in matrices:
            matrix[basis_indices, :] = 0.0
            matrix[basis_indices, basis_indices] = 1.0
        F[basis_indices] = phi
        return matrices
    
def plot_results(vertices, u, n_cells_x, n_cells_y, method="AS", subdomain ="S1", Overlap=0.1):
        if not os.path.exists('figures'):
            os.makedirs('figures')
    
        half_length = len(u) // 2
        u1 = u[:half_length]
        u2 = u[half_length:]
    
        X = vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1)
        Y = vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1)
        U1 = u1.reshape(n_cells_y+1, n_cells_x+1)
        U2 = u2.reshape(n_cells_y+1, n_cells_x+1)
    
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(X, Y, U1)
        plt.colorbar(label="U1")
        plt.title(f"U1/2 Field - Cells: {n_cells_x}x{n_cells_y}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(f"figures/{method}_{subdomain}_U_{n_cells_x}x{n_cells_y}_Overlap_{Overlap}.pdf")
        plt.show()
    
def initialize_domains(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y, overlap):

    vertices_full_domain, cells_full_domain = generate_mesh(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y)

    delta_x = (vertices_full_domain[cells_full_domain[:, 1], 0] - vertices_full_domain[cells_full_domain[:, 0], 0]).flatten()
    delta_y = (vertices_full_domain[cells_full_domain[:, 2], 1] - vertices_full_domain[cells_full_domain[:, 0], 1]).flatten()

    #Subdomain 1
    i = int((n_cells_x/2 + overlap * n_cells_x))
    x_min_S1 = 0.0
    x_max_S1 = sum(delta_x[:i])
    print(delta_x[0])
    n_cells_x_S1 = i
    vertices_S1, cells_S1 = generate_mesh(x_min_S1, x_max_S1, y_min, y_max, n_cells_x_S1, n_cells_y)

    #Subdomain 2
    j = int((n_cells_x/2 - overlap * n_cells_x))
    x_min_S2 = sum(delta_x[:j])
    x_max_S2 = 2.0
    n_cells_x_S2 = n_cells_x-j
    vertices_S2, cells_S2 = generate_mesh(x_min_S2, x_max_S2, y_min, y_max, n_cells_x_S2, n_cells_y)
    return vertices_full_domain, cells_full_domain, vertices_S1, cells_S1, vertices_S2, cells_S2, x_min_S1, x_max_S1, x_min_S2, x_max_S2, n_cells_x, n_cells_x_S1, n_cells_x_S2, n_cells_y

def update_boundary_S2(vertices_boundary_S2_in_S1, n_cells_x_S2, n_cells_y, u_S1, F2):
    i_idx = np.zeros(n_cells_y + 1, dtype=np.int64)
    j_idx = np.arange(0, n_cells_y + 1)
    basis_indices = i_idx + (n_cells_x_S2 + 1)*j_idx
    for i in range(0, len(vertices_boundary_S2_in_S1)):
        F2[basis_indices[i]] = u_S1[vertices_boundary_S2_in_S1[i]]
        F2[basis_indices[i] + int(len(F2)/2)] = u_S1[vertices_boundary_S2_in_S1[i] + vertices_S1.shape[0]]
    return F2

def update_boundary_S1(vertices_boundary_S1_in_S2, n_cells_x_S1, n_cells_y, u_S2, F1):
    # Right boundary
    i_idx = n_cells_x_S1*np.ones(n_cells_y + 1, dtype=np.int64)
    j_idx = np.arange(0, n_cells_y + 1)
    basis_indices = i_idx + (n_cells_x_S2 + 1)*j_idx
    for i in range(0, len(vertices_boundary_S1_in_S2)):
        F1[basis_indices[i]] = u_S2[vertices_boundary_S1_in_S2[i]]
        F1[basis_indices[i] + int(len(F1)/2)] = u_S2[vertices_boundary_S1_in_S2[i] + vertices_S2.shape[0]]
    return F1

def alternating_Schwarz(n_cells_x_S1, n_cells_x_S2, n_cells_y, F1, F2, S_S1, S_S2):
    method = "AS"
    vertices_boundary_S2_in_S1 = np.where(np.isclose(vertices_S1[:, 0], x_min_S2))[0]
    vertices_boundary_S1_in_S2 = np.where(np.isclose(vertices_S2[:, 0], x_max_S1))[0]
    residual_S1 = residual_S2 = 1.0

    # Initialize u
    u_S1 = np.zeros(2*vertices_S1.shape[0])
    u_S2 = np.zeros(2*vertices_S2.shape[0])
    for i, vertice in enumerate(vertices_S1):
        u_S1[i] = u_S1[i+ int(len(u_S1)/2)]= 1 * vertice[1]* (1- vertice[1])
    for i, vertice in enumerate(vertices_S2):
        u_S2[i] =u_S2[i+ int(len(u_S2)/2)]= 1 * vertice[1]* (1- vertice[1])

    F2 = update_boundary_S2(vertices_boundary_S2_in_S1, n_cells_x_S2, n_cells_y, u_S1, F2)
    F1 = update_boundary_S1(vertices_boundary_S1_in_S2, n_cells_x_S1, n_cells_y, u_S2, F1)
    i = 0
    residual_sum = 1.0
    start_time = time.time()
    while residual_sum > 1e-3:
        i += 1
        u1_preupdate = u_S1.copy()
        u_S1 = scipy.sparse.linalg.spsolve(S_S1, F1)
        residual_S1 = np.linalg.norm(u1_preupdate - u_S1)

        F2 = update_boundary_S2(vertices_boundary_S2_in_S1, n_cells_x_S2, n_cells_y, u_S1, F2)
        u2_preupdate = u_S2.copy()
        u_S2 = scipy.sparse.linalg.spsolve(S_S2, F2)
        residual_S2 = np.linalg.norm(u2_preupdate - u_S2)

        F1 = update_boundary_S1(vertices_boundary_S1_in_S2, n_cells_x_S1, n_cells_y, u_S2, F1)
        residual_sum = residual_S1 + residual_S2
    
    end_time = time.time()

    print(f"Converged in {i} iterations, Time taken: {end_time - start_time} seconds")

    plot_results(vertices_S1, u_S1, n_cells_x_S1, n_cells_y,method, "S1", overlap)
    plot_results(vertices_S2, u_S2, n_cells_x_S2, n_cells_y,method, "S2", overlap)

def parallel_Schwarz(n_cells_x_S1, n_cells_x_S2, n_cells_y, F1, F2, S_S1, S_S2):
    method = "PS"
    vertices_boundary_S2_in_S1 = np.where(np.isclose(vertices_S1[:, 0], x_min_S2))[0]
    vertices_boundary_S1_in_S2 = np.where(np.isclose(vertices_S2[:, 0], x_max_S1))[0]
    residual_S1 = residual_S2 = 1.0

    #initialize u1 and u2
    u_S1 = np.ones(2*vertices_S1.shape[0])
    u_S2 = np.ones(2*vertices_S2.shape[0])

    for i, vertice in enumerate(vertices_S1):
        u_S1[i] = u_S1[i+ int(len(u_S1)/2)]= 1 * vertice[1]* (1- vertice[1])
    for i, vertice in enumerate(vertices_S2):
        u_S2[i] =u_S2[i+ int(len(u_S2)/2)]= 1 * vertice[1]* (1- vertice[1])

    F2 = update_boundary_S2(vertices_boundary_S2_in_S1, n_cells_x_S2, n_cells_y, u_S1, F2)
    F1 = update_boundary_S1(vertices_boundary_S1_in_S2, n_cells_x_S1, n_cells_y, u_S2, F1)
    i = 0
    residual_sum = 1.0
    time_parallel = 0
    while residual_sum > 1e-3:
        i += 1
        u1_preupdate = u_S1.copy()
        u2_preupdate = u_S2.copy()

        start_time = time.time()
        u_S1 = scipy.sparse.linalg.spsolve(S_S1, F1)
        end_time = time.time()
        time_uS1 = end_time - start_time

        start_time = time.time()
        u_S2 = scipy.sparse.linalg.spsolve(S_S2, F2)
        end_time = time.time()
        time_uS2 = end_time - start_time

        if time_uS1 > time_uS2:
            time_parallel += time_uS1
        else:
            time_parallel += time_uS2

        F1 = update_boundary_S1(vertices_boundary_S1_in_S2, n_cells_x_S1, n_cells_y, u_S2, F1)
        F2 = update_boundary_S2(vertices_boundary_S2_in_S1, n_cells_x_S2, n_cells_y, u_S1, F2)

        residual_S1 = np.linalg.norm(u1_preupdate - u_S1)     
        residual_S2 = np.linalg.norm(u2_preupdate - u_S2)

        residual_sum = residual_S1 + residual_S2

    print(f"Converged in {i} iterations, Time taken: {time_parallel} seconds")

    plot_results(vertices_S1, u_S1, n_cells_x_S1, n_cells_y, method, "S1", overlap)
    plot_results(vertices_S2, u_S2, n_cells_x_S2, n_cells_y, method, "S2", overlap)

# Define domain
x_min = y_min = 0.0
x_max = 1.0
y_max = 1.0
n_cells_x = 20
n_cells_y = 20
overlap = 0.1
# Initialize domains
vertices_full_domain, cells_full_domain, vertices_S1, cells_S1, vertices_S2, cells_S2, x_min_S1, x_max_S1, x_min_S2, x_max_S2, n_cells_x, n_cells_x_S1, n_cells_x_S2, n_cells_y = initialize_domains(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y, overlap)
delta_x = (vertices_full_domain[cells_full_domain[:, 1], 0] - vertices_full_domain[cells_full_domain[:, 0], 0]).flatten()
delta_y = (vertices_full_domain[cells_full_domain[:, 2], 1] - vertices_full_domain[cells_full_domain[:, 0], 1]).flatten()
# Boundary conditions
phi_left =  phi_right = phi_bottom = phi_top = 0.0

#Forcing term
F_global = compute_Forcing(vertices_full_domain,delta_x[0],delta_y[0])
F1 = compute_Forcing(vertices_S1,delta_x[0],delta_y[0])
F2 = compute_Forcing(vertices_S1,delta_x[0],delta_y[0])

# Compute global matrices
A1_global = compute_global_A_1_A_2(vertices_full_domain, cells_full_domain)
A2_global = A1_global.copy()
A1_S1 = compute_global_A_1_A_2(vertices_S1, cells_S1)
A2_S1 = A1_S1.copy()
A1_S2 = compute_global_A_1_A_2(vertices_S2, cells_S2)
A2_S2 = A1_S2.copy()

matrices_global = [A1_global, A2_global]
matrices_S1 = [A1_S1, A2_S1]
matrices_S2 = [A1_S2, A2_S2]

# Set boundary conditions
set_boundary_conditions(n_cells_x, n_cells_y, F_global, matrices_global, phi_left, phi_right, phi_bottom, phi_top)
set_boundary_conditions(n_cells_x_S1, n_cells_y, F1, matrices_S1, phi_left, phi_right, phi_bottom, phi_top)
set_boundary_conditions(n_cells_x_S2, n_cells_y, F2, matrices_S2, phi_left, phi_right, phi_bottom, phi_top)

#Set matrices
S_S1 = scipy.sparse.bmat([[A1_S1, None], [None, A2_S1]]).tocsr()
S_S2 = scipy.sparse.bmat([[A1_S2, None], [None, A2_S2]]).tocsr()
S_global = scipy.sparse.bmat([[A1_global, None], [None, A2_global]]).tocsr()

#Solve global system for comparison
u_global = scipy.sparse.linalg.spsolve(S_global, F_global)
print(delta_x[0], delta_y[0])
print(max(u_global))
plot_results(vertices_full_domain, u_global, n_cells_x, n_cells_y, "GS", "Full", 0)

#Solve with Schwarz
#alternating_Schwarz(n_cells_x_S1, n_cells_x_S2, n_cells_y, F1, F2, S_S1, S_S2)
#parallel_Schwarz(n_cells_x_S1, n_cells_x_S2, n_cells_y, F1, F2, S_S1, S_S2)

