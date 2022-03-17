import kwant 
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.linalg as la
from numpy import sqrt
import tinyarray as ta
import matplotlib.pyplot as plt

lat_const = 1  # lattice constant of kagome (unit: nm)
t1 = -1.0    # nearest neighbor hopping parameter for kagome (unit: eV)
t2 = -1.0*0.0    # next nearest neighbor hopping parameter for kagome (unit: eV)
tn = 0.0    # interlayer hopping between kagomes (unit: eV)

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
    
    sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder(sym)

	# define hopping and on-site potential on bottom layer graphene
    bulk[B_sub_a(0,0)] = np.random.normal(0, 1, 1)
    bulk[B_sub_b(0,0)] = np.random.normal(0, 1, 1)
    bulk[B_sub_c(0,0)] = np.random.normal(0, 1, 1)
    
    bulk[kwant.builder.HoppingKind((0,0), B_sub_a,B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((0,0), B_sub_b,B_sub_c)] = t1
    bulk[kwant.builder.HoppingKind((0,0), B_sub_c,B_sub_a)] = t1
    bulk[kwant.builder.HoppingKind((-1,0), B_sub_a,B_sub_c)] = t1
    bulk[kwant.builder.HoppingKind((0,1), B_sub_c,B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((1,-1),B_sub_b,B_sub_a)] = t1
    
        # Next neighbors

    bulk[kwant.builder.HoppingKind((0,1), B_sub_c,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((1,-1), B_sub_c,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((1,0), B_sub_b,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((0,-1), B_sub_b,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((-1,0), B_sub_a,B_sub_b)] = t2
    bulk[kwant.builder.HoppingKind((0,1),B_sub_a,B_sub_b)] = t2
    bulk[kwant.builder.HoppingKind((1,0), B_sub_c,B_sub_b)] = t2
    bulk[kwant.builder.HoppingKind((-1,1), B_sub_c,B_sub_b)] = t2
    bulk[kwant.builder.HoppingKind((1,-1), B_sub_b,B_sub_c)] = t2
    bulk[kwant.builder.HoppingKind((-1,0), B_sub_b,B_sub_c)] = t2
    bulk[kwant.builder.HoppingKind((-1,1), B_sub_a,B_sub_c)] = t2
    bulk[kwant.builder.HoppingKind((0,-1),B_sub_a,B_sub_c)] = t2

	# define hopping and on-site potential on upper layer graphene
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

def square(site):
    x, y = abs(site.pos)
    return x < 100 and y < 100



for x in [0.00, 0.25, 0.33, 0.55, 0.66, 0.75, 1.00]:
    t1 = -1 + x
    t2 = -x

    syst = kwant.Builder() 
    model = make_system(type_kagome = 'monolayer')
    syst.fill(model, square, (0, 0));
    kwant.plot(syst);

    fsyst = syst.finalized()
    rho = kwant.kpm.SpectralDensity(fsyst, num_moments = 1000)
    energies, densities = rho()

    plt.plot(energies, densities)
    plt.title("Euler Kagome DOS with KPM (t = "+str(round(t1, 2))+", t' = "+str(round(t2, 2))+")")
    plt.xlim(-5,3)
    plt.ylim(0,100000)
    plt.xlabel("Energy (E) / (t + t')")
    plt.ylabel(r"Density of states $\rho(E)$ (a.u.)")
    plt.tight_layout()
    #plt.margins(x = 1.5, y = 1.0)
    #plt.show()
    plt.savefig("DOS_dis_"+str(x)+".png", dpi='figure')
    plt.clf()
 
