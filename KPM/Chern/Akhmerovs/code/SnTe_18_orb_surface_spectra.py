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
import matplotlib.pyplot as plt
import pickle

from hamiltonians import SnTe_18band_disorder, site_type, SnPbTe_params
from spectral_functions import surface_dos


# # Calculate surface spectral functions of disordered slabs with PBC in x and y

# In[ ]:


import hpc05


# In[ ]:


hpc05.kill_remote_ipcluster()


# In[ ]:


client, dview, lview = hpc05.start_remote_and_connect(11, profile='pbs_32GB', timeout=600,
                                                      env_path='~/.conda/envs/kwant_dev',
                                                      folder='~/disorder_invariants/code/',
                                                     )


# In[ ]:


get_ipython().run_cell_magic('px', '--local', 'import itertools as it\nimport scipy\nimport scipy.linalg as la\nimport numpy as np\nimport copy\nimport functools as ft\nimport pickle\nimport tinyarray as ta\n\nimport kwant\nfrom hamiltonians import SnTe_18band_disorder, site_type, SnPbTe_params\nfrom spectral_functions import surface_dos')


# In[ ]:


get_ipython().run_cell_magic('px', '--local', "salt = '0'\n\n# Make a slab with PBC in x and y\nLx = Ly = 80\nLz = 20\nLs = ta.array([Lx, Ly, Lz])\n\nnum_moments = 4000\nenergies = np.linspace(-1, 1, 1001)\n\nd = 1\n# surface layer\nwindow = lambda pos: pos[2] >= Ls[2]/2 - d + 0.5\n# slab\nshape = lambda site: -Ls[2]/2 < site.pos[2] <= Ls[2]/2\n\n@ft.lru_cache(maxsize=1)\ndef make_syst(Ls, doping, salt):\n    Lx, Ly, Lz = Ls\n    syst = SnTe_18band_disorder()\n\n    # Build film using syst\n    film = kwant.Builder(kwant.lattice.TranslationalSymmetry([Lx, 0, 0], [0, Ly, 0]))\n\n    film.fill(syst, shape, start=np.zeros(3));\n    filmw = kwant.wraparound.wraparound(film)   \n    filmw = filmw.finalized()\n\n    pars = SnPbTe_params.copy()\n\n    pars['site_type'] = ft.partial(site_type, doping=doping, n=None, to_fd=None, salt=salt)\n    pars['k_x'] = pars['k_y'] = pars['k_z'] = 0\n    \n    return filmw, pars\n\ndef job(k, n=ta.array([1, 1, 0])/np.sqrt(2)):\n    filmw, pars = make_syst(Ls, doping, salt)\n    print(k, end='\\r')\n    \n    sf = surface_dos(filmw,\n                     k * n,\n                     pars,\n                     supercell=np.diag(Ls),\n                     pos_transform=None,\n                     num_moments=num_moments,\n                     bounds=None,\n                     window=window)\n    return sf(energies)")


# In[ ]:


get_ipython().run_cell_magic('px', '--local', 'doping = 0.8')


# In[ ]:


# %%time
ks = np.linspace(1.5, np.pi/np.sqrt(2), 101)
result = lview.map_async(job, ks)


# In[ ]:


result.wait_interactive()


# In[ ]:


res = result.get()
spec_func = np.array(res).T.real
plt.figure(figsize=(20,10))
plt.imshow(spec_func,
           # vmin=0, vmax=200,
           extent=(ks[0], ks[-1], energies[0], energies[-1]),
           # aspect=1/2, 
           origin='lower',
           
           )
plt.colorbar()


# In[ ]:


pickle.dump(dict(
                doping=doping,
                spec_func=spec_func,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                salt=salt,
                num_moments=num_moments,
                ),
            open('../data/SnTe_18_orb_specfunc_'+str(Lx)+'x'+str(Ly)+'x'+str(Lz)+'_'+str(doping)+'.pickle', 'wb'))

