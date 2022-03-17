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

from hamiltonians import SnTe_6band_disorder, doped_m
from hamiltonians import SnTe_6band_params as SnTe_params
from mirror_chern import mirror_chern, make_window, pg_op, M_cubic, UM_p
from kpm_funcs import position_operator


# # Mirror chern number in 6-band model

# Start up cluster we use the `hpc05` [package](https://github.com/basnijholt/hpc05), use your favorite method to get an `ipyparallel.client.view.LoadBalancedView` object.

# In[ ]:


hpc05.kill_remote_ipcluster()


# In[ ]:


import hpc05
client, dview, lview = hpc05.start_remote_and_connect(300, profile='pbs', timeout=600,
                                                      env_path='~/.conda/envs/kwant_dev/',
                                                      folder='~/disorder_invariants/code/',
                                                     )


# In[ ]:


get_ipython().run_cell_magic('px', '--local', 'import itertools as it\nimport scipy\nimport scipy.linalg as la\nimport numpy as np\nimport copy\nimport functools as ft\n\nimport kwant\nfrom hamiltonians import SnTe_6band_disorder, doped_m\nfrom hamiltonians import SnTe_6band_params as SnTe_params\nfrom mirror_chern import mirror_chern, make_window,  pg_op, M_cubic, UM_p\nfrom kpm_funcs import position_operator')


# In[ ]:


get_ipython().run_cell_magic('px', '--local', "# Make a slab with PBC in one direction and open BC in the orthogonal directions\n# surface normal\nn = np.array([1, 1, 0])\nn11 = np.array([1, -1, 0])\nnz = np.array([0, 0, 1])\n# thickness (number of atomic layers - 1)\nW = 40\nL11 = 40\nLz = 60\n\nnum_vectors = 5\nnum_moments = 1000\n# salt specifies the disorder realization used\nsalt = '2'\n\nnum_m = 51\nm_array = np.linspace(-1, 4, num_m)\n\nnum_x = 21\nx_array = np.linspace(0, 1, num_x)\n\ndef make_operators(doping, mPb):\n    syst2 = SnTe_6band_disorder()\n\n    # Build film using syst\n    film = kwant.Builder(kwant.lattice.TranslationalSymmetry(W * n, L11 * n11, Lz * nz))\n\n    film.fill(syst2, lambda site: True, start=np.zeros(3));\n    filmw = kwant.wraparound.wraparound(film)   \n    filmw = filmw.finalized()\n\n    M_trf = ft.partial(M_cubic, n=n)\n    UM = UM_p(n)\n    M = pg_op(filmw, M_trf, UM)\n\n    pars = SnTe_params.copy()\n    mSn = SnTe_params['m'][1]\n    mTe = SnTe_params['m'][0]\n    to_fd = filmw._wrapped_symmetry.to_fd\n\n    pars['m'] = ft.partial(doped_m, doping=doping, mSn=mSn, mPb=mPb, mTe=mTe, n=n, to_fd=to_fd, salt=salt)\n    # gap should be near the weighted average\n    pars['mu'] = -(((1 - doping) * mSn + doping * mPb) + mTe) / 2\n    pars['k_x'] = pars['k_y'] = pars['k_z'] = 0\n\n    H = filmw.hamiltonian_submatrix(params=pars, sparse=True)\n    ham_size = H.shape[0]\n    norbs = filmw.sites[0].family.norbs\n\n    x0, y0, z0 = position_operator(filmw)\n    x = 1/np.sqrt(2) * (x0 - y0)\n    y = z0\n\n    # window half the size\n    win_L11 = L11//2\n    win_Lz = Lz//2\n    A = win_L11 * win_Lz * np.sqrt(2)\n\n    def shape1(site):\n        tag = site.tag\n        tag11 = np.dot(tag, n11) // np.dot(n11, n11)\n        tagz = np.dot(tag, nz) // np.dot(nz, nz)\n        tagn = (np.dot(tag, n) % (W * n.dot(n))) // np.dot(n, n)\n        return (-win_L11/2 + L11/2 < tag11 <= win_L11/2 + L11/2 and\n                -win_Lz/2 + Lz/2 < tagz <= win_Lz/2 + Lz/2 and\n                tagn < W//2)\n    window1 = make_window(filmw, shape1)\n    \n    def shape2(site):\n        tag = site.tag\n        tag11 = np.dot(tag, n11) // np.dot(n11, n11)\n        tagz = np.dot(tag, nz) // np.dot(nz, nz)\n        tagn = (np.dot(tag, n) % (W * n.dot(n))) // np.dot(n, n)\n        return (-win_L11/2 + L11/2 < tag11 <= win_L11/2 + L11/2 and\n                -win_Lz/2 + Lz/2 < tagz <= win_Lz/2 + Lz/2 and\n                tagn >= W//2)\n    window2 = make_window(filmw, shape2)\n\n    return H, M, x, y, [window1, window2], pars, A\n    \n\ndef job(doping, mPb):\n    print(doping, mPb)\n    \n    H, M, x, y, windows, pars, A = make_operators(doping, mPb)\n    \n    spectrum = kwant.kpm.SpectralDensity(H, num_moments=num_moments, params=pars)\n\n    es, dos = spectrum()\n    ran = np.logical_and(-1 < es, es < 1)\n    minimum = np.argmin(dos[ran])\n    mine = es[ran][minimum]\n    mindos = dos[ran][minimum]\n    \n    filling = spectrum.integrate(distribution_function=lambda x: x<mine) / spectrum.integrate()\n\n    C_list = [mirror_chern(H, x, y, Mz=M, vectors=num_vectors,\n                          e_F=mine, kpm_params=dict(num_moments=num_moments),\n                          params=pars, bounds=None, window=window, return_std=False)\n              for window in windows]\n\n    C_list = np.array(C_list)\n    Cs = np.sum(C_list, axis=0) / A\n    C = np.mean(Cs)\n    C_std = np.std(Cs)\n    return C, C_std, filling, mine, mindos")


# In[ ]:


xms = [(x, m) for x, m in it.product(x_array, m_array)]
result = lview.map_async(job, *zip(*xms))


# In[ ]:


result.wait_interactive()


# In[ ]:


all([err is None for err in result.error])


# In[ ]:


res_array = np.array(result.get())
MZ_vs_xm, MZ_std_vs_xm, filling, mine, mindos = res_array.T
MZ_vs_xm = MZ_vs_xm.reshape((len(x_array), len(m_array)))
MZ_std_vs_xm = MZ_std_vs_xm.reshape((len(x_array), len(m_array)))
filling = filling.reshape((len(x_array), len(m_array)))
mine = mine.reshape((len(x_array), len(m_array)))
mindos = mindos.reshape((len(x_array), len(m_array)))


# In[ ]:


import pickle
pickle.dump(dict(params=SnTe_params, 
                 x_array=x_array,
                 m_array=m_array,
                 e_F=0,
                 W=W,
                 L11=L11,
                 Lz=Lz,
                 disorder_realizations=1,
                 MZ_vs_xm=MZ_vs_xm,
                 MZ_std_vs_xm=MZ_std_vs_xm,
                 filling=filling,
                 mine=mine,
                 mindos=mindos,
                 num_vectors=num_vectors,
                 num_moments=num_moments,
                 salt=salt,
                 description=('Pb doped SnTe using a slab of SnTe_6band_disorder with PBC in all directions, sizes W perpendicular and Lz, L11 parallel to mirror planes. '
                              'Two mirror planes are included and a factor of 1/2 ommited (hence the mirror Chern number is doubled). '
                              'Mirror Chern number is calculated using mirror_chern averaged for the interior (half linear size in directions parallel to the mirror planes). '
                              'The doping concentration of Sn->Pb substitution (x_array) and Pb onsite potential (m_array) is varied. '
                              'One disorder realization is used, the std of MZ comes from different random vectors.'
                              'The chemical potential is alligned to the minimum of the DoS and filling is checked.')
                ),
            open('../data/Mirror_Chern_SnXTe_6orb_'+salt+'.pickle', 'wb'))


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(MZ_vs_xm.T.real, vmin=0, vmax=5,
           extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
           aspect=1/2, origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel(r'$m_{Pb}$')
plt.title(r'$MZ$')
plt.show()


# In[ ]:


plt.imshow(MZ_std_vs_xm.T.real,
           # vmin=0, vmax=3,
           extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
           aspect=1/2, origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel(r'$m_{Pb}$')
plt.title(r'$MZ$')


# In[ ]:


plt.imshow(mindos.T.real,
           # vmin=0, vmax=10000,
           extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
           aspect=1/2, origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel(r'$m_{Pb}$')
plt.title(r'$MZ$')


# In[ ]:


plt.imshow(filling.T.real,
           # vmin=0, vmax=3,
           extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
           aspect=1/2, origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel(r'$m_{Pb}$')
plt.title(r'$MZ$')


# In[ ]:


plt.imshow(mine.T.real,
           # vmin=0, vmax=3,
           extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
           aspect=1/2, origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel(r'$m_{Pb}$')
plt.title(r'$MZ$')

