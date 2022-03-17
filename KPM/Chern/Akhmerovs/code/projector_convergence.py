#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import kwant
import hamiltonians
import kpm_funcs
from matplotlib import pyplot as plt

import holoviews as hv
hv.notebook_extension()


# In[ ]:


params = hamiltonians.chern_params


# In[ ]:


params


# In[ ]:


L = 41
syst = hamiltonians.chern_insulator(L=L)
ham = syst.hamiltonian_submatrix(params=params)


# In[ ]:


spectrum = kwant.kpm.SpectralDensity(ham)
hv.Curve((spectrum.energies, spectrum.densities.real),
         kdims='e', vdims='DOS [a.u.]', label='Density of states')


# In[ ]:


def exact_projector(ham):
    evals, evecs = np.linalg.eigh(ham)
    evecs = evecs[:, evals < 0]
    return evecs.conj() @ evecs.T


# In[ ]:


# define local vectors

vectors = np.array([v for v in kwant.kpm.LocalVectors(syst, where=lambda s: s.tag == [L//2, L//2])])


# In[ ]:


# define kpm projector

kpm_params = dict(num_moments = 100)
projector = kpm_funcs.build_projector(ham, vectors, kpm_params)
projected_vectors = projector()


# In[ ]:


def vectors_density(vector, norbs=2):
    vector = np.linalg.norm(vector.squeeze(), axis=0)
    vector = sum([vector[i::norbs] for i in range(norbs)]) #sum over orbitals
    return vector
proj_density = vectors_density(projected_vectors)


# In[ ]:


# Plot the lattice and the density of a projected local vector 
kwant.plot(syst, site_size=0.5, site_color=proj_density, cmap='copper_r');


# In[ ]:


kpm_dens_10 = vectors_density(kpm_funcs.build_projector(ham, vectors, kpm_params=dict(num_moments=10))())
kpm_dens_20 = vectors_density(kpm_funcs.build_projector(ham, vectors, kpm_params=dict(num_moments=20))())
kpm_dens_40 = vectors_density(kpm_funcs.build_projector(ham, vectors, kpm_params=dict(num_moments=40))())


# In[ ]:


exact_projected_vector = (exact_projector(ham) @ vectors.T).T
exact_density = vectors_density(exact_projected_vector)


# In[ ]:


xcut_args = [i for i, s in enumerate(syst.sites) if s.pos[1] == L//2]


# In[ ]:


plt.rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(111)
plt.setp(ax.spines.values(), linewidth=1.5)

ax.tick_params(width=1.5)
ax.tick_params(length=4.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
x_array = np.arange(L) - L//2
plt.plot(x_array, exact_density[xcut_args], label=r'$M=\infty$', c='k', lw=4)
plt.plot(x_array, kpm_dens_10[xcut_args], label=r'$M=10$', c='C0', ls='-', lw=2)
plt.plot(x_array, kpm_dens_20[xcut_args], label=r'$M=20$', c='C2', ls='-', lw=2)
plt.plot(x_array, kpm_dens_40[xcut_args], label=r'$M=40$', c='C1', ls='-', lw=2)

plt.yscale('log')
plt.ylabel(r"$\left|\left\langle x_0, y_0\right| \hat{P} \left|x, y_0 \right\rangle\right|$")
plt.ylim(1e-6, 1)
plt.xlabel(r'$x-x_0$')

plt.legend(loc=(0.75, 0.45))
plt.tight_layout()

plt.savefig('../manuscript/figures/projector_convergence.pdf', bbox_inches = 'tight', pad_inches = 0.1)
plt.show()

