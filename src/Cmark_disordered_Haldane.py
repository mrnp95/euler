import kwant 
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.linalg as la
from numpy import sqrt
import tinyarray as ta
import matplotlib.pyplot as plt

import mirror_chern as ch
from kpm_funcs import position_operator

# Default settings

lat_const = 1  # lattice constant of graphene (unit: nm)
t1 = 1.0    # nearest neighbor hopping parameter for graphene (unit: eV)
t2 = -0.5*(1j)    # next nearest neighbor hopping parameter for graphene (unit: eV)
d  = 1.0    # standard deviation in Gaussian disorder (unit: eV)
L = 100 # size of the system (in each dimension)

disorders = [0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
averaging = 2 # number of runs for averaging DOS and conductivities

# Domains of cond function for later

N_bins = 100 # Bins for energies in the estimator
N_binsT = 500 # Bins for temperature
T_min = 0.01 
T_max = 5.00
T = np.linspace(T_min, T_max, N_binsT)

def make_system(type_graphene = 'monolayer'):
    Bravais_vector = [(    lat_const,                     0), 
                      (0.5*lat_const, 0.5*lat_const*sqrt(3))] # Bravais vectors
    Bottom_Lat_pos = [(0, lat_const/sqrt(3)), (0,  0)]    # The position of sublattice atoms in the layer  
    
    Bottom_lat = kwant.lattice.general(Bravais_vector, Bottom_Lat_pos, norbs=1)
    B_sub_a, B_sub_b = Bottom_lat.sublattices
    
    sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder()
    
    for x in range(int(-L/2), int(L/2)):
        for y in range(int(-L/2), int(L/2)):

	    # define hopping and on-site potential on bottom layer kagome

            bulk[B_sub_a(x,y)] = np.random.normal(0, d, 1)
            bulk[B_sub_b(x,y)] = np.random.normal(0, d, 1)
            #print(bulk[B_sub_a(x,y)])
    #print(bulk[B_sub_a(0,0)], bulk[B_sub_a(1,1)], B_sub_a(-1,-1))
    
    bulk[kwant.builder.HoppingKind((0,0), B_sub_a,B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((0,-1), B_sub_a,B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((1,-1), B_sub_a,B_sub_b)] = t1
    
        # Next neighbors

    bulk[kwant.builder.HoppingKind((1,0), B_sub_b,B_sub_b)] = -t2
    bulk[kwant.builder.HoppingKind((0,1), B_sub_b,B_sub_b)] = -t2
    bulk[kwant.builder.HoppingKind((1,-1), B_sub_b,B_sub_b)] = -t2 
    bulk[kwant.builder.HoppingKind((1,0), B_sub_a,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((0,1), B_sub_a,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((1,-1), B_sub_a,B_sub_a)] = t2
    
    # For debugging hoppings

    #bulk[Bottom_lat.neighbors()] = t1
    #bulk[B_sub_a.neighbors()] = t2
    #bulk[B_sub_b.neighbors()] = -t2

    return bulk

# Debugging syst for Haldane

def make_syst_topo(a=1, t=1, t2=0.5):
    lat = kwant.lattice.honeycomb(a, norbs=1, name=['a', 'b'])
    return lat

# Different geometries of the finite system

def trunc(site):
    x, y = abs(site.pos)
    return abs(x) < 800 and abs(y) < 800

def circle(site):
    x, y = site.pos
    r = 30
    return x ** 2 + y ** 2 < r ** 2

# Declare big things coming


for t in [0.50]:
    
    legend = []
    
    C_final = []

    fig_C, ax_C = plt.subplots()

    for d in disorders:
        
        print("|t'| = "+str(t)+", d = "+str(d))

        all_energies = []
        all_densities = []
        all_cond_xx_miu = []
        all_cond_xy_miu = []
        all_T = []
        all_cond_xx_T = []
        all_cond_xy_T = []

        C_av = 0

        for av in range(0, averaging):
        
            # Hoppings
            
            t1 = 1.0
            t2 = -t*(1j)

            syst = kwant.Builder() 
            model = make_system(type_graphene = 'monolayer')
            #kwant.plot(model)
            #model = make_syst_topo()
            area_per_site = np.abs(lat_const*lat_const*np.sqrt(3)/2)/2
            syst.fill(model, trunc, (0, 0));
            
            # Debugging...

            #syst[model.shape(trunc,(0,0))] = 0.
            #syst[model.neighbors()] = t1
            # add second neighbours hoppings
            #syst[model.a.neighbors()] = -t2
            #syst[model.b.neighbors()] = t2
            
            syst.eradicate_dangling()
            
            # Plot system before running
            
            #kwant.plot(syst);


            fsyst = syst.finalized()

            # Chern markers

            num_vectors = 5
            num_moments = 1000
            A = area_per_site*2*L*L

            C_list = []

            systw = kwant.wraparound.wraparound(syst)   
            fsystw = systw.finalized()
            x, y = position_operator(fsystw)
            
            # Window half the size
            
            Lx = L
            Ly = L
            nx = np.array([1, 0])
            ny = np.array([0, 1])
            win_Lx = Lx//2
            win_Ly = Ly//2

            def shape1(site):
                tag = np.array(site.tag)
                tagy = np.dot(tag, ny) // np.dot(ny, ny)
                tagx = np.dot(tag, nx) // np.dot(nx, nx)
                #tagn = (np.dot(tag, n) % (W * n.dot(n))) // np.dot(n, n)
                return (-win_Lx/2 + Lx/2 < tagx <= win_Lx/2 + Lx/2 and -win_Ly/2 + Ly/2 < tagy <= win_Ly/2 + Ly/2) # and tagn < W//2)
                                                                                    
            def shape2(site):
                tag = np.array(site.tag)
                tagx = np.dot(tag, nx) // np.dot(nx, nx)
                tagy = np.dot(tag, ny) // np.dot(ny, ny)
                #tagn = (np.dot(tag, n) % (W * n.dot(n))) // np.dot(n, n)
                return (-win_Lx/2 + Lx/2 < tagx <= win_Lx/2 + Lx/2 and -win_Ly/2 + Ly/2 < tagy <= win_Ly/2 + Ly/2) # and tagn >= W//2)
            
            window1 = ch.make_window(fsystw, shape1)
            window2 = ch.make_window(fsystw, shape2)
            windows = [window1, window2]

            C_list = [ch.mirror_chern(fsyst, x, y, Mz=None, vectors=num_vectors, e_F=0, kpm_params=dict(num_moments=num_moments), params=None, bounds=None, window=window, return_std=False) for window in windows]

            C_list = np.array(C_list) / A
            print(C_list)

            Cs = np.sum(C_list, axis = 0)
            C = np.mean(Cs)
            C_av += C
            C_std = np.std(Cs)
            print('Averaged Local Chern = '+str(C)+', Stdev of Local Chern = '+str(C_std))
        
        C_av = 1/(averaging)*C_av
        C_final.append(C_av)

    ax_C.plot(disorders, C_final)
    ax_C.set_title("Averaged local Chern number C as a function of disorder \n (t = "+str(round(t1, 2))+", t' = "+str(round(t2, 2))+")", y=1.1)
    ax_C.set_xlabel(r"Disorder strength $\sigma$")
    ax_C.set_ylabel(r"Local Chern number averaged over system")
    ax_C.set_xlim(int(disorders[0]), int(disorders[len(disorders)-1]))
    #fig_C.legend(legend, loc='upper right', bbox_to_anchor=(0.90, 0.8))       
    fig_C.tight_layout()
    save_fig_to = './'
    fig_C.savefig(save_fig_to + "Marking_Haldane_tNNN_"+str(t)+".png", bbox_inches = 'tight', dpi='figure')

