#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy
import scipy.linalg as la
import functools as ft
import matplotlib.pyplot as plt
import tinyarray as ta
import kwant
from concurrent.futures import ProcessPoolExecutor

from hamiltonians import SnTe_6band
from hamiltonians import SnTe_6band_params as SnTe_params
from mirror_chern import mirror_chern, pg_op, M_cubic, UM_p

from kpm_funcs import position_operator
plt.rc('text', usetex=True)


# In[ ]:


from matplotlib import pyplot as plt

import holoviews as hv
hv.notebook_extension()


# ## SnTe 6-band model Chern number

# ### [001] surface

# In[ ]:


# Make a slab with PBC in one direction and open BC in the orthogonal directions

# thickness (number of atomic layers - 1)
W = 20
L = 20

# normal vector to the surface
n = np.array([1, 1, 0])
symm = kwant.lattice.TranslationalSymmetry(W * n)

template = SnTe_6band()

# add vectors perpendicular to the translational symmetry
lat = next(iter(template.sites())).family # extract lattice family from the system
symm.add_site_family(lat, other_vectors=[[1,-1,0], [0, 0, 2]])
builder = kwant.Builder(symm)

def shape(site):
    tag = site.tag
    return la.norm(tag - np.dot(tag, n) * n/np.dot(n, n)) <= L

builder.fill(template, shape, start=np.zeros(3));
syst = kwant.wraparound.wraparound(builder).finalized()


# #### Define mirror operator

# In[ ]:


# the position operators are rotated with respect to the normal axis 
x0, y0, z0 = position_operator(syst)

x = 1/np.sqrt(2) * (x0 - y0)
y = z0

# Mirror with respect to the 110 normal
M_trf = ft.partial(M_cubic, n=n)
UM = UM_p(n)
M = pg_op(syst, M_trf, UM)


# In[ ]:


fig = plt.figure();
fig.set_tight_layout(True);

ax = fig.add_subplot(1, 1, 1, projection='3d');
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('z');
ax.set_title('Slab of SnTe with {} sites'.format(len(syst.sites)));
kwant.plotter.plot(syst, ax=ax, show=False);


# In[ ]:


params = SnTe_params.copy()
params['k_x'] = 0 # set PBCs, without a phase through the boundaries
params['mu'] = 0 # zero shift in the mass term
m0 = 1.65 # mass term of Sn ions

def m(site):
    a = np.sum(site.tag) % 2
    return 2 * a * m0 - m0

params['m'] = m


# In[ ]:


# calculate the density of states

spectrum = kwant.kpm.SpectralDensity(syst, params=params, num_moments=500, num_vectors=1)
plot_dos = hv.Curve((spectrum.energies, spectrum.densities.real),
                    kdims='e', vdims='DOS [a.u.]', label='Density of states')
plot_dos + plot_dos.relabel('Detail of the density of states')[-1:1]


# In[ ]:


# choose an energy resolution and set the number of moments
dE = 0.01
num_moments = int((spectrum.bounds[1] - spectrum.bounds[0]) / (dE * np.pi))
print('number of moments necessary to resolve the gap:', num_moments)
kpm_params=dict(num_moments=200)


# ### random vectors, no distinction of layers

# In[ ]:


# Define random vectors per layer parallel to the mirror plane
# inside a disk of half radius
half_width = W / 2
area = half_width ** 2 * np.pi

def disk(site):
    pos = np.array(site.pos)
    return la.norm(pos - pos.dot(n) * n/n.dot(n)) <= half_width

vector_factory = kwant.kpm.RandomVectors(syst, where=disk)
# we must define the number of vectors for this infinite generator
num_vectors = 10
vectors = [next(vector_factory) for i in range(num_vectors)]


# In[ ]:


def mirror_chern_per_vector(vector):
    return mirror_chern(syst, x, y, Mz=M, vectors=[vector], e_F=0,
                      params=params, window=None, return_std=False)


# In[ ]:


#get_ipython().run_cell_magic('time', '', '\nwith ProcessPoolExecutor(max_workers=24) as executor:\n    this_c = executor.map(mirror_chern_per_vector, vectors)\n    \nthis_c = np.array(list(this_c))')


# In[ ]:


print('mirror Chern number:', np.mean(this_c) / area)
print('std from random vectors:', np.std(this_c) / area)


# ### random vectors per layer

# In[ ]:


half_width = L / 2
area = half_width ** 2 * np.pi

num_vectors = 10

z_array = np.arange(int(n.dot(n)) * W)
vectors = []
for z in z_array:

    def disk(site):
        pos = np.array(site.pos)
        return la.norm(pos - pos.dot(n) * n/n.dot(n)) <= half_width and pos.dot(n) == z

    vector_factory = kwant.kpm.RandomVectors(syst, where=disk)
    # we must define the number of vectors for this infinite generator
    vectors.extend([next(vector_factory) for i in range(num_vectors)])
len(vectors)


# In[ ]:


#get_ipython().run_cell_magic('time', '', '\nwith ProcessPoolExecutor(max_workers=24) as executor:\n    C_list = executor.map(mirror_chern_per_vector, vectors)')


# In[ ]:


c_layer = np.array(list(C_list)).reshape(len(z_array), num_vectors)


# In[ ]:


c_mean = np.mean(c_layer, axis=1) / area
c_std = np.std(c_layer, axis=1) / area


# In[ ]:


#get_ipython().run_cell_magic('opts', "Overlay [fig_size=120 fontsize=12 legend_position='top_right']", "%%opts Spread [show_legend=True]\n\nplot_mc = (hv.Curve((z_array, c_mean), label=r'$\\rho(C_M)$') *\n hv.Spread((z_array, c_mean, c_std), label=r'$\\sigma(\\rho(C_M))$')\n).relabel('Mirror Chern number density per layer').redim(x='layer position', y=r'$\\rho(C_M)$')\nplot_mc")


# In[ ]:


half_width = L / 2
area = half_width ** 2 * np.pi


# In[ ]:


print('Mirror Chern number (sum of contributions per layer): %2.2f' % (np.sum(c_mean)))
print('std from random vectors per layer: %2.2f' % (np.sqrt(np.sum(c_std**2))))


# In[ ]:


plt.figure(figsize=(4, 3))
plt.plot(z_array, c_mean)
plt.fill_between(z_array, c_mean+c_std, c_mean-c_std, alpha=0.5)
plt.ylabel(r'$C_M(z)$')
plt.xlabel(r'$z$')
plt.tight_layout()
plt.savefig('../manuscript/figures/mirror_chern_layer.pdf')


# In[ ]:




