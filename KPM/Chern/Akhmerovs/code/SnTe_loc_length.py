#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import hpc05
import matplotlib.pyplot as plt
import scipy.optimize


# In[ ]:


hpc05.kill_remote_ipcluster()


# Start up cluster we use the `hpc05` [package](https://github.com/basnijholt/hpc05), use your favorite method to get an `ipyparallel.client.view.LoadBalancedView` object.

# In[ ]:


client, dview, lview = hpc05.start_remote_and_connect(100, profile='pbs', timeout=300,
                                                      env_path='~/.conda/envs/kwant_dev/',
                                                      folder='~/disorder_invariants/code/',
                                                     )


# In[ ]:


get_ipython().run_cell_magic('px', '--local', 'import itertools as it\nimport scipy\nimport scipy.linalg as la\nimport numpy as np\nimport copy\nimport functools as ft\n\nimport kwant\n# print(kwant.__version__)\nfrom lcao import L_matrices, lcao_term\nfrom hamiltonians import SnTe_6band_disorder #, doped_m\nfrom hamiltonians import SnTe_6band_params as SnTe_params')


# In[ ]:


get_ipython().run_cell_magic('px', '--local', "# Lead parameters to make sure it doesn't have zero velocity modes\nparsl=dict(ml=0, tl=np.array([2, 0.5, 0.5]), k_x=0, k_y=0)\n\n# Redefine `doped_m` to work here\ndef doped_m(site, doping, mSn, mPb, mTe, rng):\n    tag = site.tag\n    a = np.sum(tag) % 2\n    if a == 0:\n        return mTe\n    else:\n        ran = rng.uniform()\n        if ran < doping:\n            return mPb\n        else:\n            return mSn\n\ndef SnTe_cylinder(Lx, Ly, Lz=None, with_lead=True):\n    if Lz is None:\n        Lz = Ly\n\n    syst = SnTe_6band_disorder()\n\n    # Scattering region\n    if with_lead:\n        builder = kwant.Builder(kwant.lattice.TranslationalSymmetry(Ly * np.array([0, 1, 0]),\n                                                                    Lz * np.array([0, 0, 1])))\n        builder.fill(syst, lambda site: np.abs(site.pos[0]) <= Lx/2, start=np.zeros(3));\n    else:\n        builder = kwant.Builder(kwant.lattice.TranslationalSymmetry(Lx * np.array([1, 0, 0]),\n                                                                    Ly * np.array([0, 1, 0]),\n                                                                    Lz * np.array([0, 0, 1])))\n        builder.fill(syst, lambda site: True, start=np.zeros(3));\n\n    builder = kwant.wraparound.wraparound(builder)\n\n    # Make lead simple cubic\n    if with_lead:\n        systl = kwant.Builder(symmetry=kwant.lattice.TranslationalSymmetry(*np.eye(3)))\n        lat = kwant.lattice.cubic(norbs=6)\n        systl[lat(0, 0, 0)] = lambda site, ml: ml * np.eye(6)\n        systl[kwant.builder.HoppingKind([1, 0, 0], lat)] = lambda site1, site2, tl: tl[0] * np.eye(6)\n        systl[kwant.builder.HoppingKind([0, 1, 0], lat)] = lambda site1, site2, tl: tl[1] * np.eye(6)\n        systl[kwant.builder.HoppingKind([0, 0, 1], lat)] = lambda site1, site2, tl: tl[2] * np.eye(6)\n        lead = kwant.Builder(kwant.TranslationalSymmetry(Ly * np.array([0, 1, 0]),\n                                                         Lz * np.array([0, 0, 1]),\n                                                         np.array([-1, 0, 0])))\n        lead.fill(systl, lambda site: True, start=np.zeros(3));\n        lead = kwant.wraparound.wraparound(lead, keep=2)\n        builder.attach_lead(lead)\n        builder.attach_lead(lead.reversed())\n    \n    return builder.finalized()")


# In[ ]:


get_ipython().run_cell_magic('px', '--local', 'def localization_length_from_transmissions(lengths, avg_log_transmissions):\n    try:\n        popt, pcov = scipy.optimize.curve_fit(affine, lengths, avg_log_transmissions)\n    except RuntimeError:\n        return np.nan, np.nan\n\n    # Uncertainty in the fit slope, allow 1 stddev deviation\n    evals, evecs = np.linalg.eig(pcov)\n    u_localization_length = np.abs(np.sqrt(evals) @ evecs[0])\n\n    # Use large value for better plotting, could also use np.inf\n    large = 1.0e6\n    if popt[0] < 0:\n        localization_length = -1/popt[0]\n    elif abs(popt[0]) < u_localization_length:\n        localization_length = large\n    else:\n        localization_length = large\n        print("Warning, poor localization length fit, \\n"\n              "popt = {}, pcov = {}".format(popt, pcov))\n    \n    return localization_length, u_localization_length\n\ndef affine(x, a, b):\n    return a * x + b')


# In[ ]:


get_ipython().run_cell_magic('px', '--local', "num_m = 51\nm_array = np.linspace(-1, 4, num_m)\n\nnum_x = 21\nx_array = np.linspace(0, 1, num_x)\n\nLs = np.array([10, 15, 20])\nn_dis = 10\n\nW = 12\n\ndef job(doping, mPb):\n    print(doping, mPb)\n    pars = SnTe_params.copy()\n    mSn = SnTe_params['m'][1]\n    mTe = SnTe_params['m'][0]\n    rng = kwant._common.ensure_rng(rng=0)\n\n    pars['m'] = ft.partial(doped_m, doping=doping, mSn=mSn, mPb=mPb, mTe=mTe, rng=rng)\n    # gap should be near the weighted average\n    pars['mu'] = -(((1 - doping) * mSn + doping * mPb) + mTe) / 2\n    pars['k_x'] = pars['k_y'] = pars['k_z'] = 0\n    pars['doping'] = doping\n    pars['mPb'] = mPb\n\n    # Find gap\n    spectrum = kwant.kpm.SpectralDensity(SnTe_cylinder(max(Ls), max(Ls), with_lead=False), \n                                         num_moments=1000, params=pars)\n    es, dos = spectrum()\n    ran = np.logical_and(-1 < es, es < 1)\n    minimum = np.argmin(dos[ran])\n    mine = es[ran][minimum]\n    mindos = dos[ran][minimum]\n    filling = spectrum.integrate(distribution_function=lambda x: x<mine) / spectrum.integrate()\n\n    Ts = []\n    var_Ts = []\n    for L in Ls:\n        syst = SnTe_cylinder(L, W, with_lead=True)\n        T, var_T = disorder_averaged_log_transmission(syst, \n                                                     {**pars, **parsl}, \n                                                     n_dis, \n                                                     energy=mine,\n                                                     return_var=True)\n        Ts.append(T - 2 * np.log(W))\n        var_Ts.append(var_T)\n    # xi = localization_length_from_transmissions(Ls, Ts)\n    # return xi\n    return Ts, var_Ts, mine, filling\n\ndef disorder_averaged_log_transmission(system, params,\n                                       disorder_realizations, energy=0, return_var=False):\n    log_transmissions = [np.log(kwant.smatrix(system, energy, params=params).transmission(1, 0))\n                         for i in range(disorder_realizations)]\n    if return_var:\n        return np.mean(log_transmissions, axis=0), np.var(log_transmissions, axis=0)\n    else:\n        return np.mean(log_transmissions, axis=0)")


# In[ ]:


mws = [(x, m) for x, m in it.product(x_array, m_array)]
result = lview.map_async(job, *zip(*mws))


# In[ ]:


result.wait_interactive()


# In[ ]:


all([err is None for err in result.error])


# In[ ]:


Ts_vs_xm = np.array(result.get())
xi_vs_xm = np.array([localization_length_from_transmissions(Ls, Ts) for Ts, *_ in Ts_vs_xm])
xi_arr = xi_vs_xm[:, 0]
xi_arr = xi_arr.reshape((len(x_array), len(m_array)))


# In[ ]:


plt.imshow(xi_arr.T, 
           vmin=0, vmax=50,
           extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
           aspect=0.4, 
           origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel(r'$m_{Pb}$')
plt.title(r'$\xi$ at fixed W=12 with L up to 20')


# In[ ]:


import pickle
pickle.dump(dict(params=SnTe_params, Ls=Ls, E=0, W=10,
                 m_array=m_array, x_array=x_array, 
                 Ts=Ts_vs_xm, xi_array=xi_arr,
                 disorder_realizations=n_dis),
            open('../data/Localization_SnXTe_6orb.pickle', 'wb'))


# In[ ]:




