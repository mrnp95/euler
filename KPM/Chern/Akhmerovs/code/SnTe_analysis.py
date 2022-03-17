#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy
import scipy.linalg as la
import functools as ft
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
np.warnings.filterwarnings('ignore')
plt.rc('text', usetex=True)


# # Analyze SnPbTe 18 orbital results

# Note that in the numerical calculations the doping level $x$ is the Pb content, while in the manuscript we use $x$ for the Sn content, hence we replace $x\to 1-x$. We also use the opposite sign convention and a factor of 2 different normalization for the mirror Chern number, so we replace $C_m \to - C_M/2$.

# In[ ]:


from collections import defaultdict

x_array = np.linspace(0, 1, 26)
def get_data(file, Lz=None):
    size = file.split('SnTe18_MC_')[-1]
    data = defaultdict(list)
    salts = defaultdict(list)
    dat = pickle.load(open(file, 'rb'))
    for key, val in dat.items():
        if isinstance(key, tuple):
            doping, salt = key
        else:
            continue
        data[doping] += list(val)
        salts[doping] += list([salt])
    return data, salts


# Load data and plot $C_M$ vs. $x$ and the standard deviation of $C_M$.

# In[ ]:


directory = '../data/'
files = ['SnTe18_MC_40x80x120.pickle',
         'SnTe18_MC_30x60x92.pickle',
         'SnTe18_MC_24x48x72.pickle',
         'SnTe18_MC_20x40x60.pickle']

C_avgs = []
for file in files:
    data, salts = get_data(directory + file)
    C_avg = np.array([np.mean(data[x]) for x in x_array])
    C_avgs.append(C_avg)
    C_std = np.array([np.std(data[x]) for x in x_array])
    plt.plot(1-x_array, -C_avg/2)
    plt.fill_between(1-x_array, -(C_avg + C_std)/2, -(C_avg - C_std)/2, alpha=0.5)
    plt.title(plt.title(file.split('SnTe18_MC_')[-1].split('.')[0]))
    plt.ylabel(r'$C_M$')
    plt.xlabel(r'$x$')
    plt.show()


# ## Finite size scaling

# In[ ]:


from scipy.interpolate import interp1d
from scipy.optimize import minimize


# In[ ]:


def resample_data(data, num_samples):
    x_min = max([min(xs) for xs, _ in data])
    x_max = min([max(xs) for xs, _ in data])
    # print(x_min, x_max)
    x_samples = np.linspace(x_min, x_max, num_samples)
    resampled = []
    for x_array, Cs in data:
        resampled.append(interp1d((1-x_array), -Cs/2, bounds_error=False)(x_samples))
    return x_samples, np.array(resampled)

def loss(data):
    diffs = np.nanvar(data, ddof=1, axis=0)
    # print(diffs)
    loss = np.nanmean(diffs)
    return loss

def rescale_data(x_c, nu, data):
    rescaled = []
    for L, (x_array, dat) in data.items():
        rescaled.append([(x_array - x_c) / L**(-1/nu), dat])
    return np.array(rescaled)

def goal_function(x_c, nu, data, num_samples):
    rescaled = rescale_data(x_c, nu, data)
    _, resampled = resample_data(rescaled, num_samples)
    return loss(resampled)


# In[ ]:


# organize data
scaling_data = {L: [1 - x_array, -Cs/2] for L, Cs in zip([120, 92, 72, 60], C_avgs)}


# In[ ]:


# Fit both x_c and nu
num_samples = 1000
def f(x):
    return goal_function(*x, scaling_data, num_samples)

res = minimize(f, (0.3, 1), method='Nelder-Mead')
print(res)
x_c, nu = res['x']


# In[ ]:


# Fit x_c with nu=1 fixed
num_samples = 1000
def f(x):
    return goal_function(x, 1, scaling_data, num_samples)

res = minimize(f, (0.3), method='Nelder-Mead')
print(res)
x_c = res['x']


# In[ ]:


Ls = [120, 92, 72, 60]

plt.figure(figsize=(3, 6))
for xs, Cs in rescale_data(x_c, nu, scaling_data):
    plt.plot(xs, Cs)
plt.xlim([-40, 40])
plt.legend(['40x80x120', '30x60x92', '24x48x72', '20x40x60'])
plt.xlabel(r'$(x - x_c) W$')
plt.ylabel('Mirror Chern number')
plt.savefig('../manuscript/figures/SnPbTe_18_band_scaling.svg')


# In[ ]:


fig, ax1 = plt.subplots(figsize=(6, 4))
for C_avg in C_avgs:
    ax1.plot(1-x_array, -C_avg/2)

ax1.plot([0, 1], [0, 0], '--', c='k')
ax1.plot([0, 1], [2, 2], '--', c='k')
ax1.plot([x_c, x_c], [0, 2], '--', c='r')
ax1.legend(['40x80x120', '30x60x92', '24x48x72', '20x40x60'], loc=(0.01, 0.6))
ax1.set_xlabel('Sn content')
ax1.set_ylabel('Mirror Chern number')

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.6, 0.3, 0.25, 0.35]

ax2 = fig.add_axes([left, bottom, width, height])

for xs, Cs in rescale_data(x_c, nu, scaling_data):
    ax2.plot(xs, Cs)
ax2.set_xlim([-40, 40])
# ax2.legend(['40x80x120', '30x60x92', '24x48x72', '20x40x60'])
ax2.set_xlabel(r'$(x - x_c) L$')
ax2.set_ylabel(r'$M_C$')

plt.savefig('../manuscript/figures/SnPbTe_18_band_transition.svg')


# ## DoS

# In[ ]:


DoS_data = pickle.load(open('../data/DoS_SnPbTe18.pickle', 'rb'))
spec_vs_x = DoS_data['spec_vs_x'].real


# In[ ]:


plt.figure(figsize=(5, 5))
for i, (es, doss) in enumerate(spec_vs_x):
    plt.plot(es, doss + 10000*x_array[i], c='k')
    # plt.plot([0.13 + 0.06 * x_array[i] - 0.08 * x_array[i]**2], [10000*x_array[i]], '.', c='r')
plt.plot(0.135 + 0.09 * x_array - 0.13 * x_array**2, 10000*x_array, '-', c='r')
# plt.plot(0.05 + 0.1 * x_array - 0.0 * x_array**2, 10000*x_array, '-', c='b')
# plt.plot(0.22 + 0.08 * x_array - 0.26 * x_array**2, 10000*x_array, '-', c='b')
plt.xlim(-0.1, 0.4)
plt.ylim(-100, 12000)
plt.xlabel(r'$E$')
plt.ylabel('DoS')
plt.savefig('../manuscript/figures/SnPbTe_18_band_DoS.pdf')


# # Surface spectra

# In[ ]:


Lx = Ly = 80
Lz = 20
doping = 0.8
sf_data = pickle.load(open('../data/SnTe_18_orb_specfunc_'+str(Lx)+'x'+str(Ly)+'x'+str(Lz)+'_'+str(doping)+'.pickle', 'rb'))


# In[ ]:


# 80x80x20 doping = 0.8 num_moments=4000
energies = sf_data['energies']
ks = sf_data['ks']
spec_func1 = sf_data['spec_func']
plt.figure(figsize=(20,10))
plt.imshow(spec_func1,
           # vmin=0, vmax=200,
           extent=(ks[0], ks[-1], energies[0], energies[-1]),
           # aspect=1/2, 
           origin='lower',
           
           )
plt.colorbar()


# In[ ]:


Lx = Ly = 80
Lz = 20
doping = 0.5
sf_data = pickle.load(open('../data/SnTe_18_orb_specfunc_'+str(Lx)+'x'+str(Ly)+'x'+str(Lz)+'_'+str(doping)+'.pickle', 'rb'))


# In[ ]:


# 80x80x20 doping = 0.8 num_moments=4000
energies = sf_data['energies']
ks = sf_data['ks']
spec_func2 = sf_data['spec_func']
plt.figure(figsize=(20,10))
plt.imshow(spec_func2,
           # vmin=0, vmax=200,
           extent=(ks[0], ks[-1], energies[0], energies[-1]),
           # aspect=1/2, 
           origin='lower',
           
           )
plt.colorbar()


# In[ ]:


fig , ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True, figsize=(5, 5))
im1 = ax1.imshow(spec_func1,
           vmin=0, vmax=1000,
           extent=(ks[0], ks[-1], energies[0], energies[-1]),
           aspect=0.6, 
           origin='lower',          
           )
# fig.colorbar(im1, ax=ax1)
# ax1.set_xlabel(r'$k$')
ax1.set_title(r'$x = 0.5$')
ax1.xaxis.set_ticks([ks[0], ks[-1]])
ax1.xaxis.set_ticklabels([r'$\leftarrow \Gamma$', 'X'])
ax1.set_ylabel(r'$E$')
ax2 = plt.subplot(1, 2, 2)
im2 = ax2.imshow(spec_func2,
           vmin=0, vmax=1000,
           extent=(ks[0], ks[-1], energies[0], energies[-1]),
           aspect=0.6, 
           origin='lower',          
           )
# ax2.set_xlabel(r'$k$')
ax2.set_title(r'$x = 0.2$')
ax2.xaxis.set_ticklabels([])
ax2.xaxis.set_ticks([ks[0], ks[-1]])
ax2.xaxis.set_ticklabels([r'$\leftarrow \Gamma$', 'X'])
ax2.yaxis.set_ticklabels([])
# ax2.set_ylabel(r'$E$')
# fig.colorbar(im2)
# plt.subplots_adjust(wspace=-0.2)
plt.tight_layout()
# plt.show()
plt.savefig('../manuscript/figures/SnPbTe_18_band_surface_spectra.svg')


# In[ ]:


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

plt.rc('text', usetex=True)
fig = plt.figure(figsize=(6, 6))

gs = gridspec.GridSpec(2, 2, figure=fig,
                       width_ratios=[1, 1],
                       height_ratios=[1, 1.5],
                       # wspace=0.05,
                       )

ax1 = plt.subplot(gs[0, 1])
ax2 = plt.subplot(gs[0, 0])
ax3 = plt.subplot(gs[1, :])

im1 = ax1.imshow(spec_func1,
           vmin=0, vmax=1000,
           extent=(ks[0], ks[-1], energies[0], energies[-1]),
           aspect=0.3, 
           origin='lower',          
           )
# fig.colorbar(im1, ax=ax1)
# ax1.set_xlabel(r'$k$')
ax1.set_title(r'$x = 0.5$')
ax1.xaxis.set_ticks([ks[0], ks[-1]])
ax1.xaxis.set_ticklabels([r'$\leftarrow \Gamma$', r'X'])
ax1.yaxis.set_ticks([-1, 0, 1])
ax1.yaxis.set_ticklabels([])

im2 = ax2.imshow(spec_func2,
           vmin=0, vmax=1000,
           extent=(ks[0], ks[-1], energies[0], energies[-1]),
           aspect=0.3, 
           origin='lower',          
           )
# ax2.set_xlabel(r'$k$')
ax2.set_title(r'$x = 0.2$')
ax2.xaxis.set_ticklabels([])
ax2.xaxis.set_ticks([ks[0], ks[-1]])
ax2.xaxis.set_ticklabels([r'$\leftarrow \Gamma$', r'X'])
ax2.yaxis.set_ticks([-1, 0, 1])
ax2.set_ylabel(r'$E$')

for C_avg in C_avgs:
    ax3.plot(1-x_array, -C_avg/2)

ax3.plot([0, 1], [0, 0], '--', c='k')
ax3.plot([0, 1], [2, 2], '--', c='k')
ax3.plot([x_c, x_c], [0, 2], '--', c='r')
ax3.plot([0.2, 0.2], [0, 2], ':', c='b')
ax3.plot([0.5, 0.5], [0, 2], ':', c='b')
ax3.legend([r'40x80x120', r'30x60x92', r'24x48x72', r'20x40x60'], loc=(0.01, 0.55), framealpha=1)
ax3.set_xlabel('$x$ (Sn content)')
ax3.yaxis.set_ticks([0, 1, 2])
ax3.set_ylabel(r'$C_M$ (Mirror Chern number)')

ax4 = inset_axes(ax3, width="100%", height="100%",
                 bbox_to_anchor=(0.65, 0.25, 0.25, 0.5),
                 bbox_transform=ax3.transAxes,
                 borderpad=0)

for xs, Cs in rescale_data(x_c, nu, scaling_data):
    ax4.plot(xs, Cs)
ax4.set_xlim([-40, 40])
# ax2.legend(['40x80x120', '30x60x92', '24x48x72', '20x40x60'])
ax4.set_xlabel(r'$(x - x_c) L$')
ax4.set_ylabel(r'$C_M$')

plt.savefig('../manuscript/figures/fig1.pdf', bbox_inches = 'tight', pad_inches = 0)


# ### Plot 6-orbital model results

# In[ ]:


files = get_ipython().getoutput('ls -d ../data/Mirror_Chern_SnXTe_6orb_*')

for i, file in enumerate(files):
    data = pickle.load(open(file, 'rb'))
    # average over disorder realizations
    if i == 0:
        x_array = data['x_array']
        m_array = data['m_array']
        MZ_vs_xm = data['MZ_vs_xm']
        SnTe_params = data['params']
    else:
        MZ_vs_xm = (i * MZ_vs_xm + data['MZ_vs_xm']) / (i + 1)


# In[ ]:


plt.imshow(MZ_vs_xm.T.real/2,
           vmin=-0.5, vmax=2.5,
           extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
           aspect=1/2, origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel(r'$m_{X}$')
plt.title(r'$C_M$')
plt.plot(x_array, (2.45 - (1-x_array) * SnTe_params['m'][1]) / x_array)
plt.plot(x_array, (0.45 - (1-x_array) * SnTe_params['m'][1]) / x_array)
plt.ylim(m_array[0], m_array[-1])
plt.show()


# In[ ]:


data = pickle.load(open('../data/Localization_SnXTe_6orb.pickle', 'rb'))

x_array = data['x_array']
m_array = data['m_array']
xi_array = data['xi_array']


# In[ ]:


plt.imshow(xi_array.T, 
           vmin=0, vmax=20,
           extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
           aspect=0.4, 
           origin='lower')

plt.plot(x_array, (2.45 - (1-x_array) * SnTe_params['m'][1]) / x_array)
plt.plot(x_array, (0.45 - (1-x_array) * SnTe_params['m'][1]) / x_array)
plt.ylim(m_array[0], m_array[-1])

plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$m_{X}$')
plt.title(r'$\xi$ at fixed $W=12$ with $L$ up to 20')


# In[ ]:


import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(5, 8))

gs = gridspec.GridSpec(1, 2, figure=fig,
                       width_ratios=[1, 1],
                       height_ratios=[1],
                       # wspace=0.05,
                       )

ax1 = plt.subplot(gs[0, 1])
ax2 = plt.subplot(gs[0, 0])

im1 = ax1.imshow(MZ_vs_xm.T.real/2,
                 vmin=-0.5, vmax=2.5,
                 extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
                 aspect=1/2,
                 origin='lower')
# fig.colorbar(im1, ax=ax1)

ax1.plot(x_array, (2.45 - (1-x_array) * SnTe_params['m'][1]) / x_array)
ax1.plot(x_array, (0.45 - (1-x_array) * SnTe_params['m'][1]) / x_array)
ax1.set_ylim(m_array[0], m_array[-1])

ax1.set_title(r'$C_M$')
ax1.set_xlabel(r'$x$')
# ax1.xaxis.set_ticks([ks[0], ks[-1]])
# ax1.xaxis.set_ticklabels([r'$\leftarrow \Gamma$', 'X'])
# ax1.yaxis.set_ticks([-1, 0, 1])
ax1.set_ylabel(r'$m_{X}$')

im2 = ax2.imshow(xi_array.T.real, 
                 vmin=0, vmax=20,
                 extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
                 aspect=1/2, 
                 origin='lower')
# fig.colorbar(im2, ax=ax2)

ax2.set_title(r'$\xi$')
ax2.set_xlabel(r'$x$')
ax2.plot(x_array, (2.45 - (1-x_array) * SnTe_params['m'][1]) / x_array)
ax2.plot(x_array, (0.45 - (1-x_array) * SnTe_params['m'][1]) / x_array)
ax2.set_ylim(m_array[0], m_array[-1])

# ax1.xaxis.set_ticks([ks[0], ks[-1]])
# ax1.xaxis.set_ticklabels([r'$\leftarrow \Gamma$', 'X'])
# ax1.yaxis.set_ticks([-1, 0, 1])
ax2.set_ylabel(r'$m_{X}$')

plt.tight_layout()

# plt.savefig('../manuscript/figures/6band.pdf', bbox_inches = 'tight', pad_inches = 0)


# In[ ]:


from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rc('text', usetex=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4))

im1 = ax1.imshow(MZ_vs_xm.T.real/2,
                 vmin=-0.5, vmax=2.5,
                 extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
                 aspect='auto',
                 origin='lower')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_ticks([0, 1, 2])

ax1.plot(x_array, (2.45 - (1-x_array) * SnTe_params['m'][1]) / x_array, '#8c564b')
ax1.plot(x_array, (0.45 - (1-x_array) * SnTe_params['m'][1]) / x_array, '#8c564b')
ax1.plot([0, 1], [2.85, 2.85], '--', c='C3')
ax1.set_ylim(m_array[0], m_array[-1])

ax1.set_title(r'$C_M$')
ax1.set_xlabel(r'$x$')
ax1.xaxis.set_ticks([0, 0.5, 1])
ax1.set_ylabel(r'$m_{X}$')

im2 = ax2.imshow(xi_array.T.real, 
                 vmin=0, vmax=15,
                 extent=(x_array[0], x_array[-1], m_array[0], m_array[-1]),
                 aspect='auto',
                 origin='lower',
                 cmap='magma')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im2, cax=cax, orientation='vertical');
cbar.set_ticks([0, 10, 20])

ax2.set_title(r'$\xi$')
ax2.set_xlabel(r'$x$')
ax2.xaxis.set_ticks([0, 0.5, 1])
ax2.yaxis.set_ticklabels([])
ax2.plot(x_array, (2.45 - (1-x_array) * SnTe_params['m'][1]) / x_array, '#8c564b')
ax2.plot(x_array, (0.45 - (1-x_array) * SnTe_params['m'][1]) / x_array, '#8c564b')
ax2.plot([0, 1], [2.85, 2.85], '--', c='C3')
ax2.set_ylim(m_array[0], m_array[-1])

# ax2.set_ylabel(r'$m_{X}$')

plt.tight_layout()

plt.savefig('../manuscript/figures/phase_diag_6band.pdf', bbox_inches = 'tight', pad_inches = 0.05)

