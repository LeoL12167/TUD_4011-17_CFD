#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import os
dir_name = "C:/Users/bradl/CFD - Final Project Part 1/"
plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))


# In[3]:


def phi_0(x):
    if abs(1-x) <= 1:
        return 0.5 * (1 + np.cos(np.pi * (x - 1)))
    else:
        return 0
x1_list = np.linspace(0, 4, 800)
x2_list = [x - u for x in x1_list]
phi1_list = list(map(phi_0, x2_list))


# In[4]:


# dt = 0.01
n = 100
u = 1
L = 4

# nlist = [20, 200, 1000]
dtlist = [0.005, 0.05, 0.5]
for dt in dtlist:
# for n in nlist:
    h = L/n
    
    C = u*dt/h

    ### Crank-Nicolson / Trapezoid method
    # Define the matrices
    below_diag = np.zeros((n,n))
    for i in range(1,n):
        below_diag[i,i-1] = 1

    B = (1 + dt*u/(2*h))* np.identity(n) - dt*u/(2*h)*below_diag
    A = (1 - dt*u/(2*h))* np.identity(n) + dt*u/(2*h)*below_diag

    # Define c0
    x_list = list(np.linspace(h,L,n))
    c0 = np.zeros((n,1))
    for i, xi in enumerate(x_list):
        c0[i,0] = phi_0(xi)

    c_list = [c0]
    k_max = 1/dt
    k = 0
    ck = c0
    while k < k_max:
        k += 1
        ckp1 = np.linalg.solve(B,A@ck)

        c_list.append(ckp1)
        ck = ckp1

    # Complete the lists by adding the endpoints
    x_list.insert(0,0)
    c_at1_list = c_list[-1].tolist()
    c_at1_list.insert(0,[0])
    c0 = c0.tolist()
    c0.insert(0,[0])
    
    plt.plot(x1_list, phi1_list, linestyle ='--', label = 'Exact solution at t=1')
    plt.plot(x_list,c0, label = 't=0')
    plt.plot(x_list, c_at1_list, label = 't=1')
    
    plt.legend()
    plt.title("C = " + str(C))
    plt.xlabel('$x$')
    plt.ylabel('$\phi$')
#     plt.savefig("Exercise 1.3b - Upwind - C = " + str(C) + ".jpeg", dpi=500)
    plt.figure()


# In[ ]:


# ### SUPG-discretisation
# # the functions that are in the Gaussian form
# def w_Gauss(j, i, x_tilde):
#     if j-i-1 <= x_tilde <= j-i:
#         return x_tilde - (j-i-1)
#     elif j-1 < x_tilde <= j-i+1 and j != n:
#         return j-i - x_tilde
#     else:
#         return 0

# def w_x_Gauss(j, i, x_tilde):
#     if j-i-1 <= x_tilde <= j-i:
#         return 1/h
#     elif j-1 < x_tilde <= j-i+1 and j != n:
#         return -1/h
#     else:
#         return 0
    
# # print(w_x_Gauss(2,1,1/np.sqrt(3)))


# In[5]:


### SUPG-discretisation
# integral_type_nr = 0 if w is increasing on the interval
# integral_type_nr = 1 if w is decreasing on the interval
def w_Gauss(integral_type_nr, y):
    if integral_type_nr == 0:
        return (y+1)/2
    elif integral_type_nr == 1:
        return (1-y)/2
def w_x_Gauss(integral_type_nr, y):
    if integral_type_nr == 0:
        return 1
    elif integral_type_nr == 1:
        return -1


# In[ ]:


# n = 200
# dt = 0.01
# u = 1
# L = 4

# h = L/n

# eps = u*h/2
# tau = eps/(u**2)


# In[ ]:


# M = np.zeros((n,n))
# K = np.zeros((n,n))

# G1 = -1/np.sqrt(3)
# G2 = 1/np.sqrt(3)
# Glist = [G1, G2]

# ## Create M
# # the diagonal
# for ix in range(n-1):
#     i = ix + 1 # from 1 to n-1
#     for G in Glist:
#         M[ix,ix] += (h/2) * w_Gauss(0, G)**2
#         M[ix,ix] += (h/2) * w_Gauss(1, G)**2
#         M[ix,ix] += (h/2) * tau * u * w_x_Gauss(0, G) * w_Gauss(0, G)
#         M[ix,ix] += (h/2) * tau * u * w_x_Gauss(1, G) * w_Gauss(1, G)
# for G in Glist:
#     M[-1,-1] += (h/2) * w_Gauss(0, G)**2
#     M[-1,-1] += (h/2) * tau * u * w_x_Gauss(0, G) * w_Gauss(0, G)
# # the lower diagonal
# for ix in range(1,n):
#     i = ix + 1 # from 2 to n
#     for G in Glist:
#         M[ix,ix-1] += (h/2) * w_Gauss(0, G) * w_Gauss(1,G)
#         M[ix,ix-1] += (h/2) * tau * u * w_x_Gauss(0, G) * w_Gauss(1,G)
# # the upper diagonal
# for ix in range(n-1):
#     i = ix + 1 # from 1 to n-1
#     for G in Glist:
#         M[ix,ix+1] += (h/2) * w_Gauss(1, G) * w_Gauss(0,G)
#         M[ix,ix+1] += (h/2) * tau * u * w_x_Gauss(1, G) * w_Gauss(0,G)
        
# print(M)
# ## Create K
# # the diagonal
# for ix in range(n-1):
#     i = ix + 1 # from 1 to n-1
#     for G in Glist:
#         K[ix,ix] += (h/2) * tau * u**2 * w_x_Gauss(0, G)**2
#         K[ix,ix] += (h/2) * tau * u**2 * w_x_Gauss(1, G)**2
#         K[ix,ix] += (h/2) * u * w_Gauss(0, G) * w_x_Gauss(0, G)
#         K[ix,ix] += (h/2) * u * w_Gauss(1, G) * w_x_Gauss(1, G)
# for G in Glist:
#     K[-1,-1] += (h/2) * tau * u**2 * w_x_Gauss(0, G)**2
#     K[-1,-1] += (h/2) * u * w_Gauss(0, G) * w_x_Gauss(0, G)
# # the lower diagonal
# for ix in range(1,n):
#     i = ix + 1 # from 2 to n
#     for G in Glist:
#         K[ix,ix-1] += (h/2) * u * w_Gauss(0, G) * w_x_Gauss(1,G)
#         K[ix,ix-1] += (h/2) * tau * u**2 * w_x_Gauss(0, G) * w_x_Gauss(1,G)
# # the upper diagonal
# for ix in range(n-1):
#     i = ix + 1 # from 1 to n-1
#     for G in Glist:
#         K[ix,ix+1] += (h/2) * u * w_Gauss(1, G) * w_x_Gauss(0,G)
#         K[ix,ix+1] += (h/2) * tau * u**2 * w_x_Gauss(1, G) * w_x_Gauss(0,G)
# print(K)


# In[ ]:


# # Define the matrices
# C = np.identity(n) + dt/2 * np.linalg.inv(M)@K
# D = np.identity(n) - dt/2 * np.linalg.inv(M)@K

# ## Write d instead of c and y instead of x

# # Define d0
# y_list = list(np.linspace(h,L,n))
# d0 = np.zeros((n,1))
# for i, xi in enumerate(y_list):
#     d0[i,0] = phi_0(xi)

# d_list = [d0]
# k_max = 100/dt
# k = 0
# dk = d0
# while k < k_max:
#     k += 1
#     dkp1 = np.linalg.solve(C,D@dk)
    
#     d_list.append(dkp1)
#     dk =dkp1

# plt.plot(y_list,d0)
# plt.plot(y_list,d_list[int(k_max/2)])
# # Complete the lists by adding the endpoints
# y_list.insert(0,0)
# d_at1_list = d_list[-1].tolist()
# d_at1_list.insert(0,[0])
# plt.plot(y_list, d_at1_list)
# plt.figure()

# print("The Courant Number is " + str(u*dt/h))


# In[6]:


# dt = 0.01
n = 100
u = 1
L = 4

# nlist = [20, 200, 1000]
dtlist = [0.005, 0.05, 0.5]
for dt in dtlist:
# for n in nlist:
    h = L/n
    eps = u*h/2
    tau = eps/(u**2)
    
    C = u*dt/h
    
    ####
    M = np.zeros((n,n))
    K = np.zeros((n,n))

    G1 = -1/np.sqrt(3)
    G2 = 1/np.sqrt(3)
    Glist = [G1, G2]

    ## Create M
    # the diagonal
    for ix in range(n-1):
        i = ix + 1 # from 1 to n-1
        for G in Glist:
            M[ix,ix] += (h/2) * w_Gauss(0, G)**2
            M[ix,ix] += (h/2) * w_Gauss(1, G)**2
    for G in Glist:
        M[-1,-1] += (h/2) * w_Gauss(0, G)**2
    # the lower diagonal
    for ix in range(1,n):
        i = ix + 1 # from 2 to n
        for G in Glist:
            M[ix,ix-1] += (h/2) * w_Gauss(0, G) * w_Gauss(1,G)
            M[ix,ix-1] += (h/2) * tau * u * w_x_Gauss(0, G) * w_Gauss(1,G)
    # the upper diagonal
    for ix in range(n-1):
        i = ix + 1 # from 1 to n-1
        for G in Glist:
            M[ix,ix+1] += (h/2) * w_Gauss(1, G) * w_Gauss(0,G)
            M[ix,ix+1] += (h/2) * tau * u * w_x_Gauss(1, G) * w_Gauss(0,G)

    ## Create K
    # the diagonal
    for ix in range(n-1):
        i = ix + 1 # from 1 to n-1
        for G in Glist:
            K[ix,ix] += (h/2) * tau * u**2 * w_x_Gauss(0, G)**2
            K[ix,ix] += (h/2) * tau * u**2 * w_x_Gauss(1, G)**2
    for G in Glist:
        K[-1,-1] += (h/2) * tau * u**2 * w_x_Gauss(0, G)**2
    # the lower diagonal
    for ix in range(1,n):
        i = ix + 1 # from 2 to n
        for G in Glist:
            K[ix,ix-1] += (h/2) * u * w_Gauss(0, G) * w_x_Gauss(1,G)
            K[ix,ix-1] += (h/2) * tau * u**2 * w_x_Gauss(0, G) * w_x_Gauss(1,G)
    # the upper diagonal
    for ix in range(n-1):
        i = ix + 1 # from 1 to n-1
        for G in Glist:
            K[ix,ix+1] += (h/2) * u * w_Gauss(1, G) * w_x_Gauss(0,G)
            K[ix,ix+1] += (h/2) * tau * u**2 * w_x_Gauss(1, G) * w_x_Gauss(0,G)
    
    ####
    
    # Define the matrices
    D = np.identity(n) + dt/2 * np.linalg.inv(M)@K
    E = np.identity(n) - dt/2 * np.linalg.inv(M)@K

    ## Write d instead of c and y instead of x

    # Define d0
    y_list = list(np.linspace(h,L,n))
    d0 = np.zeros((n,1))
    for i, xi in enumerate(y_list):
        d0[i,0] = phi_0(xi)

    d_list = [d0]
    k_max = 3/dt
    k = 0
    dk = d0
    while k < k_max:
        k += 1
        dkp1 = np.linalg.solve(D,E@dk)

        d_list.append(dkp1)
        dk =dkp1

    # Complete the lists by adding the endpoints
    y_list.insert(0,0)
    d_at1_list = d_list[-1].tolist()
    d_at1_list.insert(0,[0])
    d0 = d0.tolist()
    d0.insert(0,[0])
    
    plt.plot(x1_list, phi1_list, linestyle ='--', label = 'Exact solution at t=1')
    plt.plot(y_list,d0, label = 't=0')
    plt.plot(y_list, d_at1_list, label = 't=1')
    plt.legend()
    plt.title("C = " + str(C))
    plt.xlabel('$x$')
    plt.ylabel('$\phi$')
#     plt.savefig("Exercise 1.3b - SUPG - C = " + str(C) + ".jpeg", dpi=500)
    plt.figure()


# In[ ]:


print((h/2) * w_Gauss(0, G1)**2 + (h/2) * w_Gauss(0, G2)**2)

xi = i*h
xim1 = (i-1)*h
value = 1/h**2 * (xi**3/3 - xi**2*xim1 + xi*xim1**2)
value += -1/h**2 * (xim1**3/3 - xim1**2*xim1 + xim1*xim1**2)
print(value)

