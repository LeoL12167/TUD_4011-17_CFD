import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


for i in range(2, 100):
    if 120 % i == 0:
        print(i)



########################################################################
# Load the data
data = pd.read_csv('results_overlap.csv')

# Create a new figure for the first plot
fig1, ax1 = plt.subplots()

# Plot the 3rd and 5th columns on the y-axis
ax1.plot(data.iloc[:, -1], data.iloc[:, 2], marker='o', label='PCG')
ax1.plot(data.iloc[:, -1], data.iloc[:, 4], marker='x', label='CG')
ax1.set_xlabel(data.columns[-1])
ax1.set_ylabel('Time (s)')
ax1.legend(loc='upper left')
plt.savefig('figures/overlap_plot_time.pdf')

# Create a new figure for the second plot
fig2, ax2 = plt.subplots()

# Plot the 4th and 6th columns on the y-axis
ax2.plot(data.iloc[:, -1], data.iloc[:, 3], marker='o', label="CG")
ax2.plot(data.iloc[:, -1], data.iloc[:, 1], marker='x', label="PCG")
ax2.set_xlabel(data.columns[-1])
ax2.set_ylabel('Iterations')
ax2.legend(loc='upper right')
plt.savefig('figures/overlap_plot_iterations.pdf')


################################################################################################################################################
data_cells = pd.read_csv('results_ncells.csv')

# Create a new figure for the first plot
fig1, ax1 = plt.subplots()

# Plot the 3rd and 5th columns on the y-axis
ax1.plot(data_cells.iloc[:, -2], data_cells.iloc[:, 2], marker='o', label='PCG')
ax1.plot(data_cells.iloc[:, -2], data_cells.iloc[:, 4], marker='x', label='CG')
ax1.set_xlabel(data_cells.columns[-2])
ax1.set_ylabel('Time (s)')
ax1.legend(loc='upper left')
plt.savefig('figures/ncells_plot_time.pdf')

# Create a new figure for the second plot
fig2, ax2 = plt.subplots()

# Plot the 4th and 6th columns on the y-axis
ax2.plot(data_cells.iloc[:, -2], data_cells.iloc[:, 1], marker='o', label="PCG")
ax2.plot(data_cells.iloc[:, -2], data_cells.iloc[:, 3], marker='x', label="CG")
ax2.set_xlabel(data.columns[-2])
ax2.set_ylabel('Iterations')
ax2.legend(loc='upper right')
plt.savefig('figures/ncells_plot_iterations.pdf')


########################################################################################################################################################################################################################
"""data_cells = pd.read_csv('results_subdomains.csv')

# Create a new figure for the first plot
fig1, ax1 = plt.subplots()

# Plot the 3rd and 5th columns on the y-axis
ax1.plot(data_cells.iloc[:, -3], data_cells.iloc[:, 2], marker='o', label='PCG')
ax1.plot(data_cells.iloc[:, -3], data_cells.iloc[:, 4], marker='x', label='CG')
xlabel = tuple(zip(data_cells.iloc[:, -3], data_cells.iloc[:, -2]))

ax1.set_xticks(data_cells.iloc[:, -3])
ax1.set_xticklabels(xlabel)

ax1.set_xlabel('Subdomains, Cells')


ax1.set_ylabel('Time (s)')
ax1.legend(loc='upper left')

plt.savefig('figures/subdomains_plot_time.pdf')

# Create a new figure for the second plot
fig2, ax3 = plt.subplots()

# Plot the 4th and 6th columns on the y-axis
ax3.plot(data_cells.iloc[:, -3], data_cells.iloc[:, 3], marker='o', label="CG")
ax3.plot(data_cells.iloc[:, -3], data_cells.iloc[:, 1], marker='x', label="PCG")
xlabel = tuple(zip(data_cells.iloc[:, -3], data_cells.iloc[:, -2]))

ax3.set_xticks(data_cells.iloc[:, -3])
ax3.set_xticklabels(xlabel)
ax3.set_ylabel('Iterations')
ax3.set_xlabel('Subdomains, Cells')
ax3.legend(loc='upper left')

plt.savefig('figures/subdomains_plot_iterations.pdf')"""

################################################################################################################################################
data_cells = pd.read_csv('results_sub_new.csv')

# Create a new figure for the first plot
fig1, ax1 = plt.subplots()

# Plot the 3rd and 5th columns on the y-axis
ax1.plot(data_cells.iloc[:, -3], data_cells.iloc[:, 2], marker='o', label='PCG')
ax1.plot(data_cells.iloc[:, -3], data_cells.iloc[:, 4], marker='x', label='CG')
ax1.set_xlabel(data_cells.columns[-3])
ax1.set_ylabel('Time (s)')
ax1.legend(loc='upper left')
plt.savefig('figures/sub_new_time.pdf')

# Create a new figure for the second plot
fig2, ax2 = plt.subplots()

# Plot the 4th and 6th columns on the y-axis
ax2.plot(data_cells.iloc[:, -3], data_cells.iloc[:, 1], marker='o', label="PCG")
#ax2.plot(data_cells.iloc[:, -3], data_cells.iloc[:, 3], marker='x', label="CG")
ax2.set_xlabel(data.columns[-3])
ax2.set_ylabel('Iterations')
ax2.legend(loc='upper right')
plt.savefig('figures/sub_new_iter.pdf')

