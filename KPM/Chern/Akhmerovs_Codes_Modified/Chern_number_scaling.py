#!/usr/bin/env python
# coding: utf-8

# ## Chern number scaling
# 
# We analyze the scaling of the stochastic trace scaling with varying correlation length and the system size together. We use the k.p model of a simple 2-band Chern insulator without disorder, and discretize it with different lattice constants. This corresponds to the same low-energy Hamiltonian, but varying the length scale of the underlying lattice.

# In[ ]:


from matplotlib import pyplot as plt
import kwant
import tinyarray as ta
import warnings
import numpy as np
import itertools
import kpm_funcs
import scipy.linalg as la
import kwant.continuum
import pickle


# In[ ]:


def build_cont_system(a, L=10):
    ham = '(m + t1 * (k_x**2 + k_y**2)) * sigma_z + t2 * (k_x * sigma_x + k_y * sigma_y)'
    syst = kwant.continuum.discretize(ham, grid=a)
    fsyst = kwant.Builder()
    shape = lambda site: all([-L/2 < x <= L/2 for x in site.pos])
    fsyst.fill(syst, shape, (0, 0))
    return fsyst.finalized()


# ### Projector matrix element scaling

# In[ ]:


def local_vector(syst, tag, polarization=None):
    if polarization is None:
        polarization = np.ones((syst.sites[0].family.norbs))
    polarization = np.array(polarization)
    return np.concatenate([(1 if s.tag == tag else 0) * polarization for s in syst.sites])

def proj_matrix_elements(syst, params=dict(), kpm_params=None, e_F=0, proj_center=None, polarization1=None, polarization2=None, sum=True):
    if kpm_params is None:
        kpm_params = dict(num_moments=150)
    if polarization1 is None:
        polarization1 = np.eye(syst.sites[0].family.norbs)
    if polarization2 is None:
        polarization2 = np.eye(syst.sites[0].family.norbs)
    if proj_center is None:
        ham = syst.hamiltonian_submatrix(params=params, sparse=True)
        center_vecs = np.array([local_vector(syst, (0, 0), polarization) for polarization in polarization1])
        proj_center = kpm_funcs.projector(ham, center_vecs.T, kpm_params=kpm_params, e_F=e_F).T

    xs = np.unique([site.tag[0] for site in syst.sites])
    
    mx_elements = np.array([proj_center.conj() @ np.array([local_vector(syst, (x, 0), polarization) for polarization in polarization1]).T for x in xs])
    if sum:
        overlaps = np.sum(np.abs(mx_elements), (1, 2))
    else:
        overlaps = np.array(mx_elements)
        
    xs = np.unique([site.pos[0] for site in syst.sites])
    return xs, overlaps


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Takes about 10 minutes\nparams = dict(m=-1, t1=1, t2=1)\nkpm_params = dict(num_moments=2000)\n\nres = []\nL = 40\na_array = np.array([1, 0.5, 0.2, 0.1])\n\nfor a in a_array:\n    print(a)\n    syst = build_cont_system(a, L=L)\n    xs, overlaps = proj_matrix_elements(syst, params, e_F=0, kpm_params=kpm_params)\n    res.append([xs, overlaps])')


# In[ ]:


for xs, overlaps in res:
    plt.plot(xs, np.log(np.abs(overlaps)))
plt.legend(a_array)


# ### Chern operator matrix element scaling

# In[ ]:


import functools as ft
from kpm_funcs import projector

def make_xy_op(syst, pos_transform=None):
    if pos_transform is None:
        pos_transform = lambda x: x

    params = dict()
    x_op = kwant.operator.Density(syst, onsite=lambda site: pos_transform(site.pos)[0] * np.eye(site.family.norbs))
    x_op = x_op.bind(params=params)
    y_op = kwant.operator.Density(syst, onsite=lambda site: pos_transform(site.pos)[1] * np.eye(site.family.norbs))
    y_op = y_op.bind(params=params)

    return x_op, y_op

def chern_operator(syst, xy_op, vector, params=None, kpm_params=None, e_F=0,
                 bounds=None):
    """Chern operator acting on `vector`
    """
    if isinstance(syst, kwant.system.System):
        ham = syst.hamiltonian_submatrix(params=params, sparse=True).tocsr()
    else:
        ham = syst

    x_op, y_op = xy_op
    x_op, y_op = x_op.tocoo(), y_op.tocoo()
    x = lambda vec: x_op @ vec
    y = lambda vec: y_op @ vec
    pp = ft.partial(projector, kpm_params=kpm_params, ham=ham, e_F=e_F, bounds=bounds)
    p = lambda vector: pp(vectors=vector)

    p_vector = p(vector)
    pypxp_vector = p(y(p(x(p_vector))))
    pxpyp_vector = p(x(p(y(p_vector))))

    return - 2j * np.pi * (pypxp_vector - pxpyp_vector)

def chern_matrix_elements(syst, params=dict(), kpm_params=None, e_F=0, C_center=None,
                             polarization1=None, polarization2=None, sum=True):
    if kpm_params is None:
        kpm_params = dict(num_moments=150)
    if polarization1 is None:
        polarization1 = np.eye(syst.sites[0].family.norbs)
    if polarization2 is None:
        polarization2 = np.eye(syst.sites[0].family.norbs)

    if C_center is None:
        center_vecs = np.array([local_vector(syst, (0, 0), polarization)
                                for polarization in polarization1])
        ham = syst.hamiltonian_submatrix(params=params, sparse=True)
        xy_op = make_xy_op(syst)
        C_center = chern_operator(ham, xy_op, center_vecs.T, kpm_params=kpm_params, e_F=e_F)
    xs = np.unique([site.tag[0] for site in syst.sites])

    mx_elements = np.array([C_center.conj().T @ np.array([local_vector(syst, (x, 0), polarization)
                                                           for polarization in polarization2]).T for x in xs])

    if sum:
        overlaps = np.sum(np.abs(mx_elements), (1, 2))
    else:
        overlaps = mx_elements

    xs = np.unique([site.pos[0] for site in syst.sites])
    return xs, overlaps


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Takes about 30 minutes\nkpm_params = dict(num_moments=5000)\nL = 40\n\nparams = dict(m=-1, t1=1, t2=1)\na_array = np.array([1, 0.5, 0.2, 0.1])\n\nres_c = []\n\nfor a in a_array:\n    print(a)\n    syst = build_cont_system(a, L=L)\n    xs, C_overlaps = chern_matrix_elements(syst, params, e_F=0, kpm_params=kpm_params)\n    C_overlaps /= a**2\n    res_c.append([xs, C_overlaps])')


# In[ ]:


pickle.dump(dict(params=params, 
                 a_array=a_array,
                 e_F=0,
                 L=L,
                 res_c=np.array(res_c),
                 num_moments=kpm_params['num_moments'],
                ),
            open('../data/chern_operator_scaling.pickle', 'wb'))


# ### Make plot

# In[ ]:


data = pickle.load(open('../data/chern_operator_scaling.pickle', 'rb'))
a_array = data['a_array']
res_c = data['res_c']


# In[ ]:


plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(5, 3))
for xs, overlaps in res_c:
    plt.plot(xs, np.abs(overlaps))

plt.legend(a_array)
plt.xlabel(r"$\left(x-x'\right)/\xi$")
plt.ylabel(r"$\left|\left\langle x, y\right| \hat{C} \left|x', y \right\rangle\right|$")
plt.yscale('log')

plt.tight_layout()
plt.savefig('../manuscript/figures/chern_scaling.pdf')


# In[ ]:




