#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import os
dir_name = "C:/Users/bradl/CFD - Final Project Part 1/"
plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))


# In[19]:


def phi_0(x):
    if abs(1-x) <= 1:
        return 0.5 * (1 + np.cos(np.pi * (x - 1)))
    else:
        return 0


# In[23]:


# Het plotten van de gewone initiële oplossing
x_list = np.linspace(0, 4, 800)
phi_list = list(map(phi_0, x_list))
plt.plot(x_list, phi_list)
plt.xlabel('$x$')
plt.ylabel('$\phi$')
plt.savefig('Exercise 1.3a - Initial Condition.jpeg', dpi = 500)
plt.figure()


# In[24]:


u = 1
x1_list = [x - u for x in x_list]
phi1_list = list(map(phi_0, x1_list))
plt.plot(x_list, phi_list, linestyle = '--', label = 't = 0')
plt.plot(x_list, phi1_list, label = 't = 1')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$\phi$')
plt.savefig('Exercise 1.3a - Evolution t = 1.jpeg', dpi = 500)
plt.figure()

