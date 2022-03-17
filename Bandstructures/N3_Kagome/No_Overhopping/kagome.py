import kwant 
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.linalg as la
from numpy import sqrt
import tinyarray as ta
import matplotlib.pyplot as plt

lat_const = 1  # lattice constant of kagome (unit: nm)
t1 = -0.25    # nearest neighbor hopping parameter for kagome (unit: eV)
t2 = -0.25    # next nearest neighbor hopping parameter for kagome (unit: eV)
t3 = 0        # next next nearest neighbor hopping parameter for kagome (unit: eV)
tn = 0.0    # interlayer hopping between kagomes (unit: eV)

def make_system(type_kagome = 'monolayer'):
    
    Bravais_vector = [(    lat_const,                     0), 
                      (0.5*lat_const, 0.5*lat_const*sqrt(3))] # Bravais vectors
    Bottom_Lat_pos = [(0.5*lat_const, 0), (0.25*lat_const, 0.25*lat_const*sqrt(3)), (0*lat_const, 0*lat_const)]    # The position of sublattice atoms in Bottom layer  
    Upper_Lat_pos  = [(0, 0), (-0.5*lat_const, 0), (-0.25*lat_const, -0.25*lat_const*sqrt(3))]       # The position of sublattice atoms in Upper layer
    
    Bottom_lat = kwant.lattice.general(Bravais_vector, Bottom_Lat_pos, norbs=1)
    B_sub_a, B_sub_b, B_sub_c = Bottom_lat.sublattices
    
    if type_kagome == 'bilayer':
        Upper_lat  = kwant.lattice.general(Bravais_vector, Upper_Lat_pos , norbs=1)
        U_sub_a, U_sub_b, U_sub_c = Upper_lat.sublattices   
    
    sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder(sym)

	# define hopping and on-site potential on bottom layer graphene
    bulk[B_sub_a(0,0)] = 0
    bulk[B_sub_b(0,0)] = 0
    bulk[B_sub_c(0,0)] = 0
    
    bulk[kwant.builder.HoppingKind((0,0), B_sub_b,B_sub_a)] = t1
    bulk[kwant.builder.HoppingKind((0,0), B_sub_c,B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((0,0), B_sub_a,B_sub_c)] = t1
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


    #bulk[kwant.builder.HoppingKind((1,0), B_sub_a, B_sub_a)] = t3
    bulk[kwant.builder.HoppingKind((0,1), B_sub_a, B_sub_a)] = t3
    #bulk[kwant.builder.HoppingKind((-1,1), B_sub_a, B_sub_a)] = t3
    bulk[kwant.builder.HoppingKind((1,0), B_sub_b, B_sub_b)] = t3
    #bulk[kwant.builder.HoppingKind((0,1), B_sub_b, B_sub_b)] = t3
    #bulk[kwant.builder.HoppingKind((-1,1), B_sub_b, B_sub_b)] = t3 
    #bulk[kwant.builder.HoppingKind((1,0), B_sub_c, B_sub_c)] = t3
    #bulk[kwant.builder.HoppingKind((0,1), B_sub_c, B_sub_c)] = t3
    bulk[kwant.builder.HoppingKind((-1,1), B_sub_c, B_sub_c)] = t3
    

	# define hopping and on-site potential on upper layer graphene
    if type_kagome == 'bilayer': 
        bulk[U_sub_a(0,0)] = 0
        bulk[U_sub_b(0,0)] = 0
        bulk[U_sub_c(0,0)] = 0
        
        bulk[kwant.builder.HoppingKind((0,0), U_sub_a,U_sub_b)] = t1
        bulk[kwant.builder.HoppingKind((0,0), U_sub_b,U_sub_c)] = t1
        bulk[kwant.builder.HoppingKind((0,0), U_sub_c,U_sub_a)] = t1
        bulk[kwant.builder.HoppingKind((-1,0), U_sub_a,U_sub_c)] = t1
        bulk[kwant.builder.HoppingKind((0,1), U_sub_c,U_sub_b)] = t1
        bulk[kwant.builder.HoppingKind((1,-1), U_sub_b,U_sub_a)] = t1
    
        # Next neighbors

        bulk[kwant.builder.HoppingKind((0,1), U_sub_c,U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((1,-1), U_sub_c,U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((1,0), U_sub_b,U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((0,-1), U_sub_b,U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((-1,0), U_sub_a,U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((0,1), U_sub_a,U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((1,0), U_sub_c,U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((-1,1), U_sub_c,U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((1,-1), U_sub_b,U_sub_c)] = t2
        bulk[kwant.builder.HoppingKind((-1,0), U_sub_b,U_sub_c)] = t2
        bulk[kwant.builder.HoppingKind((-1,1), U_sub_a,U_sub_c)] = t2
        bulk[kwant.builder.HoppingKind((0,-1), U_sub_a,U_sub_c)] = t2
        
        #Interlayer
        
        bulk[kwant.builder.HoppingKind((0,0), U_sub_a,B_sub_b)] = tn
        bulk[kwant.builder.HoppingKind((0,0), U_sub_b,B_sub_c)] = tn
        bulk[kwant.builder.HoppingKind((0,0), U_sub_c,B_sub_a)] = tn
        bulk[kwant.builder.HoppingKind((0,0), U_sub_a,B_sub_c)] = tn
        bulk[kwant.builder.HoppingKind((0,0), U_sub_b,B_sub_a)] = tn
        bulk[kwant.builder.HoppingKind((0,0), U_sub_c,B_sub_b)] = tn
    
    return bulk

def First_BZ(finalized_bulk):
    # columns of B are lattice vectors
    B = np.array(finalized_bulk._wrapped_symmetry.periods).T
    # columns of A are reciprocal lattice vectors
    A = np.linalg.pinv(B).T

    # Get lattice points that neighbor the origin, in basis of lattice vectors
    reduced_vecs, transf = kwant.linalg.lll.lll(A.T)
    neighbors = ta.dot(kwant.linalg.lll.voronoi(reduced_vecs), transf)
    lat_ndim, space_ndim = finalized_bulk._wrapped_symmetry.periods.shape

    # Add the origin to these points.
    klat_points = np.concatenate(([[0] * lat_ndim], neighbors))

    # Transform to cartesian coordinates and rescale.
    # Will be used in 'outside_bz' function, later on.
    klat_points = 2 * np.pi * np.dot(klat_points, A.T)

    # Calculate the Voronoi cell vertices
    vor = scipy.spatial.Voronoi(klat_points)
    around_origin = vor.point_region[0]
    bz_vertices = vor.vertices[vor.regions[around_origin]]
    return vor, bz_vertices



def symm_point(text:str):
    if    text == 'G': return np.array([0,0])
    elif  text == 'K': return bz_vertices[4]
    elif  text == 'M': return 0.5 * (bz_vertices[4] +bz_vertices[5])

def k_path(*args):
    dummy_1 = list(args[0])
    dummy_2 = dummy_1[1:]
    points = zip(dummy_1[:-1], dummy_2)
                                    
    k = []
    for p1, p2 in points:
        point1 = symm_point(p1)
        point2 = symm_point(p2)
        kx=np.linspace(point1[0],point2[0],50)
        ky=np.linspace(point1[1],point2[1],50)
                                                                                        
        k.append(np.array(list(zip(kx,ky))))
                                                                                                    
    return np.concatenate(k)

def momentum_to_lattice(syst, k):
    B = np.array(syst._wrapped_symmetry.periods).T
    A = np.linalg.pinv(B).T
    k, residuals = scipy.linalg.lstsq(A, k)[:2]
    if np.any(abs(residuals) > 1e-7):
        raise RuntimeError("Requested momentum doesn't correspond"
                                " to any lattice momentum.")
    return k

def ham(sys, k_x, k_y=None, **params):
    # transform into the basis of reciprocal lattice vectors
    k = momentum_to_lattice(sys, [k_x] if k_y is None else [k_x, k_y])
    p = dict(zip(sys._momentum_names, k), **params)
    return sys.hamiltonian_submatrix(params=p, sparse=False)

def k_path_bandstructure(sys, *args):
    k_paths = k_path(args)
    energy = [] 
    for kx, ky in k_paths:
        energy.append(np.sort(np.real(np.linalg.eig(ham(sys, kx,ky))[0])))
                                
    dummy  = np.linspace(0, len(args) - 1, len(energy))

    plt.figure(figsize=(10,5))
    plt.xticks(list(range(0,len(args))), list(args))
    plt.title("|t| = "+str("{:.2f}".format(np.abs(t1)))+", |t'| = "+str("{:.2f}".format(np.abs(t2)))+", |t''| = "+str("{:.2f}".format(np.abs(t3))), y=1.03)    
    plt.xlabel("k")
    plt.ylabel("Energy / |t + t'|")
    plt.xlim(0, len(args) - 1)
    for n in range(len(args)):
        plt.axvline(x = list(range(0,len(args)))[n], color='black', linestyle = "--", linewidth = 1)
    for n in (np.array(energy)).T: 
        plt.plot(dummy, n)
    plt.savefig("Section"+str("{:.2f}".format(-x))+".png")
    #plt.show()


for x in np.linspace(0.00, 1.00, 101): 
    
    t1 = -0.25
    t2 = -0.25
    t3 = -x

    monolayer_kagome = make_system(type_kagome="monolayer")
    #bilayer_kagome   = make_system(type_kagome="bilayer")
    
    #kwant.plot(monolayer_kagome)
    
    #kwant.plot(bilayer_kagome)

    monolayer_wrapped = kwant.wraparound.wraparound(monolayer_kagome).finalized()
    #bilayer_wrapped = kwant.wraparound.wraparound(bilayer_kagome).finalized() 
    
    vor, bz_vertices = First_BZ(monolayer_wrapped)

    par = {"|t|": str("{:.2f}".format(np.abs(t1))), "|t'|": str("{:.2f}".format(np.abs(t2))), "|t''|": str("{:.2f}".format(np.abs(t3)))}
    ax = plt.figure()
    ax = kwant.wraparound.plot_2d_bands(monolayer_wrapped, params = par, show = False)
    #ax.title(str(par), y = 1.3)
    ax.savefig('BS'+str("{:.2f}".format(-x))+'.png')
    #kwant.wraparound.plot_2d_bands(bilayer_wrapped)
    k_path_bandstructure(monolayer_wrapped, 'G', 'K', 'M','G')
    #k_path_bandstructure(bilayer_wrapped  , 'G', 'K', 'M','G')
    
                                                                                        
