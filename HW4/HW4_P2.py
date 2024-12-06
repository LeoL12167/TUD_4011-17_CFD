import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
import time
from functions import *
import math


def structured_grid_decomposition(grid_size, num_subdomains, overlap, delta):

    # We want to be able to choose our overlap freely in the settings. This following part ensures, that the overlap does not cut through a cell.
    # We make sure that the overlap is at least one cell for any overlap > 0.

    # Convert overlap percentage to absolute overlap length
    overlap = overlap * grid_size / 100
    
    # Calculate the number of overlap cells
    if round(overlap / delta) > 0:
        overlap_cells = round(overlap / delta)
    else:
        overlap_cells = math.ceil(overlap / delta)
    
    # Recalculate the actual overlap distance to match the number of cells
    overlap = overlap_cells * delta
    
    # Initialize list to store subdomains and subdomain dictionaries
    subdomains = [] # Just needed for plotting
    subdomain_dicts = [] # Needed for the rest of the code

    # Calculate the size of each subdomain
    subdomain_size = grid_size / num_subdomains 

    # Loop over the number of subdomains in both x and y directions
    for i in range(num_subdomains):
        for j in range(num_subdomains):
            # Calculate the starting and ending coordinates for each subdomain in x and y directions
            x_start = max(i * subdomain_size, 0)
            x_end = min((i + 1) * subdomain_size, grid_size)
            y_start = max(j * subdomain_size, 0)
            y_end = min((j + 1) * subdomain_size, grid_size)
            
            # Adjust the coordinates by the overlap amount, ensuring they stay within grid boundaries
            x_start = max(x_start - overlap, 0)
            x_end = min(x_end + overlap, grid_size)
            y_start = max(y_start - overlap, 0)
            y_end = min(y_end + overlap, grid_size)

            # Calculate the number of cells in x and y directions for this subdomain
            cells_x = round((x_end - x_start) / delta)
            cells_y = round((y_end - y_start) / delta)

            # Create a dictionary for the subdomain's parameters
            single_subdomain_dict = {
                "x_min": x_start,
                "y_min": y_start,
                "x_max": x_end,
                "y_max": y_end,
                "n_cells_x": cells_x,
                "n_cells_y": cells_y,
            }

            # Generate mesh for the subdomain and add the details to the dictionary
            single_subdomain_dict["vertices"], single_subdomain_dict["cells"], single_subdomain_dict["delta"] = generate_mesh(single_subdomain_dict)

            # Append the dictionary to the list of subdomain dictionaries
            subdomain_dicts.append(single_subdomain_dict)

            # Append the subdomain's coordinates to the list of subdomains
            subdomains.append((x_start, x_end, y_start, y_end))

    # Return the list of subdomains and their dictionaries
    return subdomains, subdomain_dicts

def plot_grid_decomposition(subdomains, domain, n):
    #function to plot the grid decomposition, just for visualization purposes
    x_start, x_end, y_start, y_end = subdomains[0]

    subdomain_size = x_end - x_start

    grid_size = subdomains[-1][1]
    fig, ax = plt.subplots()
    
    colors = plt.cm.get_cmap('hsv', len(subdomains))
    
    for idx, subdomain in enumerate(subdomains[::n]):
        x_start, x_end, y_start, y_end = subdomain
        rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                                 linewidth=2, edgecolor=colors(idx), facecolor='none')
        ax.add_patch(rect)
        cx = (x_start + x_end) / 2
        cy = (y_start + y_end) / 2
        ax.text(cx, cy, str(idx + 1), color=colors(idx), ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size, domain["delta"] ))
    ax.set_yticks(np.arange(0, grid_size, domain["delta"] ))

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_aspect('equal')
    ax.grid(True)
    plt.title('Structured Grid Decomposition with Overlap')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Save the plot
    filename = f"figures/grid_{grid_size}_overlap_{overlap/100}_subdomain_{round(subdomain_size,2)}.pdf"
    plt.savefig(filename)

def construct_restriction_operator(domain, local_vertices):
    # Extract inner vertices of the domain
    global_vertices = domain["inner_vertices"]

    # Get the number of local and global vertices
    num_local = len(local_vertices)
    num_global = len(global_vertices)

    # Initialize a sparse matrix for the restriction operator in LIL format
    R = lil_matrix((num_local, num_global))

    # Build a KD-tree from the global vertices
    tree = cKDTree(global_vertices)

    for local_idx, local_vertex in enumerate(local_vertices):
        # Find the index of the closest global vertex to the local vertex
        global_idx = tree.query(local_vertex, k=1)[1]

        # Check if the local vertex is close to the global vertex within a tolerance
        if np.allclose(local_vertex, global_vertices[global_idx], atol=1e-5):
            # Set the corresponding entry in the restriction matrix to 1
            R[local_idx, global_idx] = 1
        # Create a mask for rows with non-zero entries
        non_zero_row_mask = R.getnnz(axis=1) > 0

    # Filter out the rows without non-zero entries (needed because the subdomains are generated before elimination of the bc)
    R = R[non_zero_row_mask]
    
    # Convert the matrix to Compressed Sparse Row format for efficient arithmetic operations
    return R.tocsr()

def construct_all_R(domain, subdomain_dicts):
    # Iterate over each subdomain dictionary
    for i, subdomain in enumerate(subdomain_dicts):
        # Construct the restriction operator for the current subdomain
        # and store it in the subdomain dictionary under the key "R"
        subdomain_dicts[i]["R"] = construct_restriction_operator(domain, subdomain["vertices"])
    
    # Return the updated list of subdomain dictionaries
    return subdomain_dicts

def decompose_stiffness_matrices_and_forcingterm(A, F, subdomain_dicts):
    # Iterate over each subdomain dictionary
    for i, subdomain in enumerate(subdomain_dicts):
        # Retrieve the restriction operator for the current subdomain
        R = subdomain["R"]
        
        # Compute the local stiffness matrix by projecting A using R
        A_local = R @ A @ R.T
        
        # Compute the local forcing term by projecting F using R
        F_local = R @ F
        
        # Store the local stiffness matrix and local forcing term in the subdomain dictionary
        subdomain_dicts[i]["A_local"] = A_local
        subdomain_dicts[i]["F_local"] = F_local

    # Return the updated list of subdomain dictionaries
    return subdomain_dicts

def additive_schwarz_preconditioner():
    # Define the preconditioner function
    def preconditioner(r, subdomain_dicts):
        # Initialize the preconditioned vector with zeros
        z = np.zeros_like(r)
        times = []
        # Iterate over each subdomain dictionary
        for dict in subdomain_dicts:
            time_loss = time.time()
            # Retrieve the local stiffness matrix and restriction operator
            A_i = dict["A_local"]
            R_i = dict["R"]
            
            # Solve the local system A_i * z_local = R_i * r
            z_local = sp.linalg.spsolve(A_i, R_i @ r)
            
            # Project the local solution back to the global system
            z_local = R_i.T @ z_local
            
            # Accumulate the contributions from each subdomain
            z += z_local
            time_loss = time.time() - time_loss
            times.append(time_loss)
        parallel_time_loss = sum(times)- max(times) 
        # Return the preconditioned vector
        return z, parallel_time_loss
    # Return the preconditioner function
    return preconditioner

def no_preconditioner():
    # Define the no_pre function
    def no_pre(r, subdomain_dicts):
        # Return the input residual vector no preconditioning, used for comparison
        return r, 0
    
    # Return the no_pre function
    return no_pre

def pcg(A, F, M, subdomain_dicts, tol=1e-6, max_iter=100):

    if not is_positive_definite_sparse(A):
        raise ValueError("Matrix A is not SPD.")
    # Initialize the solution vector with zeros (initial guess u_0)
    u = np.zeros_like(F)
        # Compute the initial residual r_0 = F - A * u_0
    r = F - A @ u
    # Apply the preconditioner to the residual to get the initial search direction
    
    z, parallel_time = M(r, subdomain_dicts)
    loss = parallel_time
    p = z
    # Compute the initial dot product of r and z
    rz_old = np.dot(r, z)
    # Iterate up to max_iter times
    for k in range(max_iter):
        # Compute the matrix-vector product A * p
        Kp = A @ p
        # Compute the step size alpha
        alpha = rz_old / np.dot(p, Kp)
        # Update the solution vector
        u = u + alpha * p
        # Update the residual vector
        r = r - alpha * Kp
        # Check the stopping criterion
        if np.linalg.norm(r) < tol:
            break
        # Apply the preconditioner to the new residual
        z, parallel_time = M(r, subdomain_dicts)
        loss += parallel_time
        # Compute the new dot product of r and z
        rz_new = np.dot(r, z)
        # Compute the step size beta
        beta = rz_new / rz_old
        # Update the search direction
        p = z + beta * p
        # Update rz_old for the next iteration
        rz_old = rz_new
    # Return the final solution and the number of iterations
    return u, k, loss

# Settings
x_min = y_min = 0.0
x_max = y_max = 1.0

# We stick to a quadratic domain. By choosing the number of cells in a smart way we ensure an easy domain decomposition in n x n subdomains.

random_factor = [10]
num_subdomains = [2, 3 ]#, 4,5,6,8,10,12,15,20,24,30,40,60]
overlap = [15] #np.arange(5,51,5) #np.arange(5, 31, 1)  # n% overlap

#We will work with many subdomins, so we use Dictioniares to store the data

def main(x_min, y_min, x_max, y_max, overlap, num_subdomains):
    n_cells_x = n_cells_y = random_factor * num_subdomains #120

    gridsize = (x_max - x_min) * (y_max - y_min)
    #Global Domain 
    domain = {
    "x_min": x_min,
    "y_min": y_min,
    "x_max": x_max,
    "y_max": y_max,
    "n_cells_x": n_cells_x,
    "n_cells_y": n_cells_y,
    "overlap": overlap
    }
    domain["vertices"], domain["cells"], domain["delta"] = generate_mesh(domain)

    # Boundary and Inner Indices of the global Domain 
    boundary_indices = compute_boundary_indices(domain["n_cells_x"], domain["n_cells_y"])
    total_indices = range(domain["vertices"].shape[0])

    domain["boundary_indices"] = sorted(list(set(boundary_indices)))

    domain["inner_indices"] = [idx for idx in total_indices if idx not in domain["boundary_indices"]]
    domain["inner_vertices"] = [v for i, v in enumerate(domain["vertices"]) if i not in domain["boundary_indices"]]
    
    #Stiffness Matrix and Forcing Term of the global Domain
    domain["A"] = compute_global_A_1_A_2(domain["vertices"], domain["cells"])
    domain["F"] = compute_Forcing(domain["vertices"], domain["cells"])
    
    #Elimination of the Boundary Conditions
    domain["A_reduced"], domain["F_reduced"] = set_boundary_with_elimination(domain["n_cells_x"], domain["n_cells_y"], domain["A"], domain["F"])

    #Decomposition of the Domain into Subdomains and construction of the Restriction Operator
    subdomains, subdomain_dicts = structured_grid_decomposition(gridsize, num_subdomains, domain["overlap"], domain["delta"])

    subdomain_dicts = construct_all_R(domain, subdomain_dicts)
    subdomain_dicts = decompose_stiffness_matrices_and_forcingterm(domain["A_reduced"], domain["F_reduced"], subdomain_dicts)

    #Apply the Preconditioner and solve the system
    time_pcg = time.time()
    u_withPre, k_PCG, parallel_time_loss = pcg(domain["A_reduced"], domain["F_reduced"], additive_schwarz_preconditioner(), subdomain_dicts)
    time_pcg = time.time() - time_pcg - parallel_time_loss
    #Comare the results with the solution without preconditioner
    time_cg = time.time()
    u_noPre, k_CG, _ = pcg(domain["A_reduced"], domain["F_reduced"], no_preconditioner(), subdomain_dicts)
    time_cg = time.time() - time_cg
    
    return {
        'k_PCG': k_PCG,
        'time_pcg': time_pcg,
        'k_CG': k_CG,
        'time_cg': time_cg,
        'num_subdomains': num_subdomains,
        'n_cells': n_cells_x**2,
        'overlap': overlap
    }

results = []
for random_factor in random_factor:
    print("random_factor", random_factor)
    for num_sub in num_subdomains:
        print("num_subdomains",num_sub)
        for overla in overlap:
            print("overlap", overla)

            result = main(x_min, y_min, x_max, y_max,  overla, num_sub)

            results.append(result)

df = pd.DataFrame(results)
df.to_csv("results.csv")


if False: #plot velocity for comparison
    #Solve the system for a reference solution
    u = sp.linalg.spsolve(domain["A_reduced"], domain["F_reduced"])
    #plot the velocity
    plot_velocity(u, domain, f"velocity_reference_{domain['n_cells_x']**2}_cells.pdf")
    plot_velocity(u_withPre, domain,  f"velocity_additiveS_{domain["n_cells_x"]**2}_cells.pdf")
    plot_velocity(u_withPre, domain,  f"velocity_no_preconditioner_{domain["n_cells_x"]**2}_cells.pdf")
    plot_grid_decomposition(subdomains, domain, 1)

