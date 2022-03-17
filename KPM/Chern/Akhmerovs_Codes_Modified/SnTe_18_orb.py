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
from inspect import getsource
print(kwant.__version__)


from lcao import L_matrices, lcao_term
from hamiltonians import sigmas


# # Calculations for 18 orbital SnPbTe

# In[ ]:


import hpc05


# In[ ]:


hpc05.kill_remote_ipcluster()


# In[ ]:


client, dview, lview = hpc05.start_remote_and_connect(26, profile='pbs_32GB', timeout=600,
                                                      env_path='/home/dvarjas/.conda/envs/kwant_dev',
                                                      folder='~/disorder_invariants/code/',
                                                      # kill_old_ipcluster=False,
                                                     )


# In[ ]:


get_ipython().run_cell_magic('px', '--local', "import itertools as it\nimport scipy\nimport scipy.linalg as la\nimport numpy as np\nimport copy\nimport functools as ft\nimport pickle\n\n# from scipy.integrate import cumtrapz\n# from scipy.interpolate import interp1d\n# from scipy.optimize import brentq\n\nimport kwant\nprint(kwant.__version__)\npickle.dump(kwant.__version__, open('version.pickle', 'wb'))\n# assert kwant.__version__ == '1.4.0a1.dev57+g402a0ca'\nfrom hamiltonians import SnTe_18band_disorder, site_type, SnPbTe_params\nfrom mirror_chern import mirror_chern, make_window, random_phase_vecs, pg_op, M_cubic, UM_spd\nfrom kpm_funcs import position_operator")


# In[ ]:


get_ipython().run_cell_magic('px', '--local', "# Make a slab with PBC\n# surface normal\nn = np.array([1, 1, 0])\nn11 = np.array([1, -1, 0])\nnz = np.array([0, 0, 1])\n# thickness (number of atomic layers - 1)\nW = 40\nL11 = 80\nLz = 120\n\nnum_vectors = 5\nnum_moments = 5000\n\nsalts = range(6, 12)\nnum_x = 26\nx_array = np.linspace(0, 1, num_x)\n\n# Set Fermi level based on half filling. Integration is inaccurate unless \n# many moments are used in the spectrum, which is expensive.\n# def set_fermi_level(spectrum, filling):\n#     energies = spectrum.energies\n#     cum_DoS = cumtrapz(spectrum(energies), energies, initial=0)\n#     total = spectrum.integrate()\n#     f = interp1d(energies, cum_DoS - total * filling, kind='quadratic')\n#     return brentq(f, min(energies), max(energies))\n\ndef set_fermi_level(doping):\n    # Set Fermi level manually, see DoS plot.\n    return 0.135 + 0.09 * doping - 0.13 * doping**2\n\n\ndef make_operators(doping, salt='salt'):\n    syst = SnTe_18band_disorder()\n\n    # Build film using syst\n    film = kwant.Builder(kwant.lattice.TranslationalSymmetry(W * n, L11 * n11, Lz * nz))\n\n    film.fill(syst, lambda site: True, start=np.zeros(3));\n    filmw = kwant.wraparound.wraparound(film)   \n    filmw = filmw.finalized()\n\n    M_trf = ft.partial(M_cubic, n=n)\n    UM = UM_spd(n)\n    M = pg_op(filmw, M_trf, UM)\n\n    pars = SnPbTe_params.copy()\n    # to_fd = filmw._wrapped_symmetry.to_fd\n\n    # window half the size\n    win_L11 = L11//2\n    win_Lz = Lz//2\n    A = win_L11 * win_Lz * np.sqrt(2)\n\n    # Make it periodic with the size of the window\n    to_fd = kwant.lattice.TranslationalSymmetry(W * n, win_L11 * n11, win_Lz * nz).to_fd\n\n    pars['site_type'] = ft.partial(site_type, doping=doping, n=n, to_fd=to_fd, salt=salt)\n    pars['k_x'] = pars['k_y'] = pars['k_z'] = 0\n\n    H = filmw.hamiltonian_submatrix(params=pars, sparse=True)\n    ham_size = H.shape[0]\n    norbs = filmw.sites[0].family.norbs\n\n    x0, y0, z0 = position_operator(filmw)\n    x = 1/np.sqrt(2) * (x0 - y0)\n    y = z0\n\n    def shape1(site):\n        tag = site.tag\n        tag11 = np.dot(tag, n11) // np.dot(n11, n11)\n        tagz = np.dot(tag, nz) // np.dot(nz, nz)\n        tagn = (np.dot(tag, n) % (W * n.dot(n))) // np.dot(n, n)\n        return (-win_L11/2 + L11/2 < tag11 <= win_L11/2 + L11/2 and\n                -win_Lz/2 + Lz/2 < tagz <= win_Lz/2 + Lz/2 and\n                tagn < W//2)\n    window1 = make_window(filmw, shape1)\n    \n    def shape2(site):\n        tag = site.tag\n        tag11 = np.dot(tag, n11) // np.dot(n11, n11)\n        tagz = np.dot(tag, nz) // np.dot(nz, nz)\n        tagn = (np.dot(tag, n) % (W * n.dot(n))) // np.dot(n, n)\n        return (-win_L11/2 + L11/2 < tag11 <= win_L11/2 + L11/2 and\n                -win_Lz/2 + Lz/2 < tagz <= win_Lz/2 + Lz/2 and\n                tagn >= W//2)\n    window2 = make_window(filmw, shape2)\n\n    return H, M, x, y, [window1, window2], pars, A\n    \n\ndef job(doping, salt='salt'):\n    salt = str(salt)\n    print(doping)\n    \n    H, M, x, y, windows, pars, A = make_operators(doping, salt)\n    \n    # spectrum = kwant.kpm.SpectralDensity(H, num_moments=num_moments, params=pars)\n    # filling = 5/18\n    # e_F = set_fermi_level(spectrum, filling)\n    e_F = set_fermi_level(doping)\n    # filling = spectrum.integrate(lambda x: x < e_F) / spectrum.integrate()\n    # mindos = spectrum(e_F)\n\n    C_list = [mirror_chern(H, x, y, Mz=M, vectors=num_vectors,\n                          e_F=e_F, kpm_params=dict(num_moments=num_moments),\n                          params=pars, bounds=None, window=window, return_std=False)\n              for window in windows]\n\n    C_list = np.array(C_list)\n    Cs = np.sum(C_list, axis=0) / A\n    C = np.mean(Cs)\n    C_std = np.std(Cs)\n    pickle.dump(dict(params=SnPbTe_params, \n                 doping=doping,\n                 e_F=e_F,\n                 W=W,\n                 L11=L11,\n                 Lz=Lz,\n                 salt=salt,\n                 Cs=Cs,\n                 num_vectors=num_vectors,\n                 num_moments=num_moments,),\n            open('disorder_invariants/data/SnTe18_MC_'+str(W)+'x'+str(L11)+'x'+str(Lz)\n                 +'/Mirror_Chern_doped_SnTe_18band_eff_'+str(salt)+'_'+str(doping)+'.pickle', 'wb'))\n    print(C, C_std)\n    return C, C_std, # filling, e_F, mindos")


# In[ ]:


# omit combinations that were already used
salts = range(0, 7)
_, existing = get_data('data/SnTe18_MC_'+str(W)+'x'+str(L11)+'x'+str(Lz)+'.pickle')
xss = [(x, s) for s, x in it.product(salts, x_array) if not str(s) in existing[x]]
len(xss)


# In[ ]:


result = lview.map_async(job, *zip(*xss))


# In[ ]:


result.wait_interactive()


# In[ ]:


all([err is None for err in result.error])


# ### Organize results

# In[ ]:


directories = get_ipython().getoutput('ls -d ../data/SnTe18_MC_*/')

for directory in directories:
    # ! rsync -rP hpc05:disorder_invariants/code/{directory} data/
    size = directory.split('SnTe18_MC_')[-1]
    size = size.split('/')[0]
    data = defaultdict(list)
    try:
        old_data = pickle.load(open(directory[:-1] + '.pickle', 'rb'))
    except:
        pass
    else:
        data.update(old_data)
    num_moments = None
    files = get_ipython().getoutput('ls {directory}Mirror_Chern_doped_SnTe_18band_eff*')
    for file in files:
        dat = pickle.load(open(file, 'rb'))
        W, L11, Lz = size.split('x')
        doping = dat['doping']
        salt = dat['salt']
        if not (dat['L11'] == int(L11) and dat['L11'] == int(L11) and dat['Lz'] == int(Lz) and
                (num_moments is None or dat['num_moments'] == num_moments) and
                np.isclose(dat['e_F'], set_fermi_level(doping))):
            assert False
        Cs = list(dat['Cs'])
        # print(directory, (doping, salt) )
        if (not (doping, salt) in data) or (not Cs[0] in data[(doping, salt)]):
            data[(doping, salt)] += Cs
        num_moments = dat['num_moments']
    data['num_moments'] = num_moments
    data['set_fermi_level'] = getsource(set_fermi_level)
    data['W'] = W
    data['L11'] = L11
    data['Lz'] = Lz
    pickle.dump(data, open(directory[:-1] + '.pickle', 'wb'))


# ## DoS

# In[ ]:


get_ipython().run_cell_magic('px', '--local', "# Make a slab with PBC\n# surface normal\nn = np.array([1, 1, 0])\nn11 = np.array([1, -1, 0])\nnz = np.array([0, 0, 1])\n\n# thickness (number of atomic layers - 1)\nW = 40\nL11 = 40\nLz = 60\n\nnum_vectors = 5\nnum_moments = 5000\n\nnum_x = 26\nx_array = np.linspace(0, 1, num_x)\n\ndef job2(doping):\n    print(doping)\n    \n    syst = SnTe_18band_disorder()\n\n    # Build film using syst\n    film = kwant.Builder(kwant.lattice.TranslationalSymmetry(W * n, L11 * n11, Lz * nz))\n\n    film.fill(syst, lambda site: True, start=np.zeros(3));\n    filmw = kwant.wraparound.wraparound(film)   \n    filmw = filmw.finalized()\n\n    pars = SnPbTe_params.copy()\n    to_fd = filmw._wrapped_symmetry.to_fd\n\n    pars['site_type'] = ft.partial(site_type, doping=doping, n=n, to_fd=to_fd, salt='salt')\n    pars['k_x'] = pars['k_y'] = pars['k_z'] = 0\n    \n    spectrum = kwant.kpm.SpectralDensity(filmw, num_moments=num_moments, params=pars)\n\n    return spectrum()")


# In[ ]:


result = lview.map_async(job2, x_array)


# In[ ]:


result.wait_interactive()


# In[ ]:


all([err is None for err in result.error])


# In[ ]:


spec_vs_x = np.array(result.get())


# In[ ]:


pickle.dump(dict(params=SnPbTe_params,
                 x_array=x_array,
                 W=W, L11=L11, Lz=Lz,
                 num_moments=num_moments,
                 disorder_realizations=1,
                 spec_vs_x=spec_vs_x,
                 ),
           open('../data/DoS_SnPbTe18.pickle', 'wb'))


# In[ ]:


plt.figure(figsize=(5, 5))
for i, (es, doss) in enumerate(spec_vs_x):
    plt.plot(es, doss.real + 10000*x_array[i], c='k')
    # plt.plot([0.13 + 0.06 * x_array[i] - 0.08 * x_array[i]**2], [10000*x_array[i]], '.', c='r')
plt.plot(0.135 + 0.09 * x_array - 0.13 * x_array**2, 10000*x_array, '-', c='r')
# plt.plot(0.05 + 0.1 * x_array - 0.0 * x_array**2, 10000*x_array, '-', c='b')
# plt.plot(0.22 + 0.08 * x_array - 0.26 * x_array**2, 10000*x_array, '-', c='b')
plt.xlim(-0.1, 0.4)
plt.ylim(-100, 12000)

