import kwant 
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.linalg as la
from numpy import sqrt
import tinyarray as ta
import matplotlib.pyplot as plt


# Default settings

lat_const = 1  # lattice constant of kagome (unit: nm)
t1 = -1.0    # nearest neighbor hopping parameter for kagome (unit: eV)
t2 = -1.0*0.0    # next nearest neighbor hopping parameter for kagome (unit: eV)
tn = 0.0    # interlayer hopping between kagomes (unit: eV)
d  = 1.0    # standard deviation in Gaussian disorder (unit: eV)
L = 100 # size of the system (in each dimension)
averaging = 5 # number of runs for averaging DOS and conductivities

# Domains of cond function for later

N_bins = 100 # Bins for energies in the estimator
T = 0.01
#disorders = [0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 3.0]
disorders = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 3.0]

############
# BUILDER  #
############

def make_system(type_kagome = 'monolayer'):
    
    Bravais_vector = [(    lat_const,                     0), 
                      (0.5*lat_const, 0.5*lat_const*sqrt(3))] # Bravais vectors
    Bottom_Lat_pos = [(0.5*lat_const, 0), (0.25*lat_const, 0.25*lat_const*sqrt(3)), (0,  0)]    # The position of sublattice atoms in Bottom layer  
    Upper_Lat_pos  = [(0, 0), (-0.5*lat_const, 0), (-0.25*lat_const, -0.25*lat_const*sqrt(3))]       # The position of sublattice atoms in Upper layer
    
    Bottom_lat = kwant.lattice.general(Bravais_vector, Bottom_Lat_pos, norbs=1)
    B_sub_a, B_sub_b, B_sub_c = Bottom_lat.sublattices
    
    if type_kagome == 'bilayer':
        Upper_lat  = kwant.lattice.general(Bravais_vector, Upper_Lat_pos , norbs=1)
        U_sub_a, U_sub_b, U_sub_c = Upper_lat.sublattices   
    
    #sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder()
    
    for x in range(int(-L/2), int(L/2)):
        for y in range(int(-L/2), int(L/2)):

	    # define hopping and on-site potential on bottom layer kagome

            bulk[B_sub_a(x,y)] = np.random.normal(0, d, 1)
            bulk[B_sub_b(x,y)] = np.random.normal(0, d, 1)
            bulk[B_sub_c(x,y)] = np.random.normal(0, d, 1)
    
    bulk[kwant.builder.HoppingKind((0,0), B_sub_a,B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((0,0), B_sub_b,B_sub_c)] = t1
    bulk[kwant.builder.HoppingKind((0,0), B_sub_c,B_sub_a)] = t1
    bulk[kwant.builder.HoppingKind((-1,0), B_sub_a,B_sub_c)] = t1
    bulk[kwant.builder.HoppingKind((0,1), B_sub_c,B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((1,-1),B_sub_b,B_sub_a)] = t1
    
        # Next neighbors

    bulk[kwant.builder.HoppingKind((0,1), B_sub_c,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((1,-1), B_sub_c,B_sub_a)] = t2
    #bulk[kwant.builder.HoppingKind((1,0), B_sub_b,B_sub_a)] = t2
    #bulk[kwant.builder.HoppingKind((0,-1), B_sub_b,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((-1,0), B_sub_a,B_sub_b)] = t2
    bulk[kwant.builder.HoppingKind((0,1),B_sub_a,B_sub_b)] = t2
    #bulk[kwant.builder.HoppingKind((1,0), B_sub_c,B_sub_b)] = t2
    #bulk[kwant.builder.HoppingKind((-1,1), B_sub_c,B_sub_b)] = t2
    bulk[kwant.builder.HoppingKind((1,-1), B_sub_b,B_sub_c)] = t2
    bulk[kwant.builder.HoppingKind((-1,0), B_sub_b,B_sub_c)] = t2
    #bulk[kwant.builder.HoppingKind((-1,1), B_sub_a,B_sub_c)] = t2
    #bulk[kwant.builder.HoppingKind((0,-1),B_sub_a,B_sub_c)] = t2

    # define hopping and on-site potential on upper layer kagome
    
    if type_kagome == 'bilayer': 
        bulk[U_sub_a(0,0)] = np.random.normal(0, 1, 1)
        bulk[U_sub_b(0,0)] = np.random.normal(0, 1, 1)
        bulk[U_sub_c(0,0)] = np.random.normal(0, 1, 1)
        
        bulk[kwant.builder.HoppingKind((0,0), B_sub_a,B_sub_b)] = t1
        bulk[kwant.builder.HoppingKind((0,0), B_sub_b,B_sub_c)] = t1
        bulk[kwant.builder.HoppingKind((0,0), B_sub_c,B_sub_a)] = t1
        bulk[kwant.builder.HoppingKind((-1,0), B_sub_a,B_sub_c)] = t1
        bulk[kwant.builder.HoppingKind((0,1), B_sub_c,B_sub_b)] = t1
        bulk[kwant.builder.HoppingKind((1,-1),B_sub_b,B_sub_a)] = t1
    
        # Next neighbors

        bulk[kwant.builder.HoppingKind((0,1), U_sub_c,U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((1,-1), U_sub_c, U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((1,0), U_sub_b, U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((0,-1), U_sub_b, U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((-1,0), U_sub_a, U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((0,1), U_sub_a, U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((1,0), U_sub_c, U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((-1,1), U_sub_c, U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((1,-1), U_sub_b, U_sub_c)] = t2
        bulk[kwant.builder.HoppingKind((-1,0), U_sub_b, U_sub_c)] = t2
        bulk[kwant.builder.HoppingKind((-1,1), U_sub_a, U_sub_c)] = t2
        bulk[kwant.builder.HoppingKind((0,-1), U_sub_a, U_sub_c)] = t2
 
        bulk[kwant.builder.HoppingKind((0,0), U_sub_a,B_sub_b)] = tn
        bulk[kwant.builder.HoppingKind((0,0), U_sub_b,B_sub_c)] = tn
        bulk[kwant.builder.HoppingKind((0,0), U_sub_c,B_sub_a)] = tn
        bulk[kwant.builder.HoppingKind((0,0), U_sub_b,B_sub_a)] = tn
        bulk[kwant.builder.HoppingKind((0,0), U_sub_c,B_sub_b)] = tn
        bulk[kwant.builder.HoppingKind((0,0), U_sub_a,B_sub_c)] = tn 
    
    return bulk

# Different geometries of the finite system

def trunc(site):
    x, y = abs(site.pos)
    return abs(x) < 800 and abs(y) < 800

def circle(site):
    x, y = site.pos
    r = 30
    return x ** 2 + y ** 2 < r ** 2

# Declare big things coming


for x in [0.00]:       
    
    legend = []

    for L in [50, 100, 150]:
        
        est_cond_xx_d = [0] * len(list(disorders))
        
        for i in list(disorders): 

            j = int(i)

            for av in range(0, averaging):
        
                # Hoppings
            
                t1 = -1.0 + x
                t2 = -x
                d = int(disorders[j])

                syst = kwant.Builder() 
                model = make_system(type_kagome = 'monolayer')
                area_per_site = np.abs(lat_const*lat_const*np.sqrt(3)/2)/3
                syst.fill(model, trunc, (0, 0));
            
                syst.eradicate_dangling()
            
                # Plot system before running
            
                #kwant.plot(syst);

                fsyst = syst.finalized()
        
                # Evaluate DOS
        
                rho = kwant.kpm.SpectralDensity(fsyst)
                energies, densities = rho()
                print("Averaging:", av+1,"/", averaging)
            
                # Evaluate conductivity

                where = lambda s: np.linalg.norm(s.pos) < 1

                # xx component
        
                s_factory = kwant.kpm.LocalVectors(fsyst, where)
                cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x', num_vectors=None, vector_factory=s_factory)
                #cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x')
            
                est_cond_xx_d[j] += (1/averaging)*cond_xx(mu = -1, temperature = T)/(area_per_site) 
            
    ############
    # Plotting #
    ############
       
        # Conductivity tensor xx component

        plt.plot(disorders, est_cond_xx_d)
        legend.append("L = "+str(L))
    
    plt.title("Average conductance $G$ as a function of disorder strength \n (t = "+str(round(t1, 2))+", t' = "+str(round(t2, 2))+")", y=1.1)
    plt.xlabel(r"Disorder strength ($\sigma$) / |t+t'|")
    plt.ylabel(r"Conductance $\langle G \rangle$ $(e^2/h)$")
    plt.xlim(int(disorders[0]), int(disorders[-1]))
    plt.tight_layout()
    plt.legend(legend)
    save_fig_to = '/mnt/d/Masters/KPM/Scaling/Kagome/'
    plt.savefig(save_fig_to + "Cond"+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
    plt.clf()
