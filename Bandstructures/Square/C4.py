import kwant 
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.linalg as la
from numpy import sqrt
import tinyarray as ta
import matplotlib.pyplot as plt

lat_const = 1  # lattice constant of square lattice (unit: nm)
tA = 0.35    # hoppings (unit: eV)
tB = 0.46
tC = 0.69    
epsA = 0
epsB = 0
epsC = 0

def make_system(type_C4 = 'monolayer'):
    
    Bravais_vector = [(lat_const, 0), 
                        (0,  lat_const)] # Bravais vectors
    Bottom_Lat_pos = [(0, 0), (0, 0), (0, 0)]    # The position of sublattice atoms in Bottom layer  
    Bottom_lat = kwant.lattice.general(Bravais_vector, Bottom_Lat_pos, norbs=1)
    B_sub_a, B_sub_b, B_sub_c = Bottom_lat.sublattices 
    
    sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder(sym)

    # define hoppings and on-site potentials
    
    bulk[B_sub_a(0,0)] = epsA
    bulk[B_sub_b(0,0)] = epsB
    bulk[B_sub_c(0,0)] = epsC
    
    bulk[kwant.builder.HoppingKind((0,1), B_sub_a,B_sub_a)] = 2*tA
    bulk[kwant.builder.HoppingKind((0,-1), B_sub_a,B_sub_a)] = 2*tA
    bulk[kwant.builder.HoppingKind((1,0), B_sub_b,B_sub_b)] = 2*tA
    bulk[kwant.builder.HoppingKind((-1,0), B_sub_b,B_sub_b)] = 2*tA
    
    bulk[kwant.builder.HoppingKind((0,1), B_sub_c,B_sub_c)] = -2*tA
    bulk[kwant.builder.HoppingKind((0,-1), B_sub_c,B_sub_c)] = -2*tA
    bulk[kwant.builder.HoppingKind((1,0), B_sub_c,B_sub_c)] = -2*tA
    bulk[kwant.builder.HoppingKind((-1,0), B_sub_c,B_sub_c)] = -2*tA
     
    bulk[kwant.builder.HoppingKind((2,0), B_sub_a,B_sub_a)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((-2,0), B_sub_a,B_sub_a)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((0,2), B_sub_a,B_sub_a)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((0,-2), B_sub_a,B_sub_a)] = -3/2*tA
    
    bulk[kwant.builder.HoppingKind((2,0), B_sub_b,B_sub_b)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((-2,0), B_sub_b,B_sub_b)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((0,2), B_sub_b,B_sub_b)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((0,-2), B_sub_b,B_sub_b)] = -3/2*tA

    bulk[kwant.builder.HoppingKind((2,0), B_sub_c,B_sub_c)] = 3*tA
    bulk[kwant.builder.HoppingKind((-2,0), B_sub_c,B_sub_c)] = 3*tA
    bulk[kwant.builder.HoppingKind((0,2), B_sub_c,B_sub_c)] = 3*tA
    bulk[kwant.builder.HoppingKind((0,-2), B_sub_c,B_sub_c)] = 3*tA
 
    bulk[kwant.builder.HoppingKind((1,1), B_sub_a,B_sub_b)] = -1/2*tB
    bulk[kwant.builder.HoppingKind((-1,-1), B_sub_a,B_sub_b)] = -1/2*tB
    bulk[kwant.builder.HoppingKind((1,-1), B_sub_a,B_sub_b)] = 1/2*tB
    bulk[kwant.builder.HoppingKind((-1,1), B_sub_a,B_sub_b)] = 1/2*tB
    
    bulk[kwant.builder.HoppingKind((2,0), B_sub_a,B_sub_c)] = tC
    bulk[kwant.builder.HoppingKind((-2,0), B_sub_a,B_sub_c)] = -tC
    bulk[kwant.builder.HoppingKind((0,2), B_sub_b,B_sub_c)] = tC
    bulk[kwant.builder.HoppingKind((0,-2), B_sub_b,B_sub_c)] = -tC

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
    elif  text == 'K': return np.array([np.pi,np.pi])
    elif  text == 'M': return np.array([0.0,np.pi])

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
    plt.title("|t$_A$| = "+str("{:.2f}".format(tA))+", |t$_B$| = "+str("{:.2f}".format(tB))+", |t$_C$| = "+str("{:.2f}".format(tC)), y=1.03)    
    plt.xlabel("k")
    plt.ylabel("Energy")
    plt.xlim(0, len(args) - 1)
    for n in range(len(args)):
        plt.axvline(x = list(range(0,len(args)))[n], color='black', linestyle = "--", linewidth = 1)
    for n in (np.array(energy)).T: 
        plt.plot(dummy, n)
    plt.savefig("Section"+str("{:.2f}".format(x))+".png")
    #plt.show()


for x in np.linspace(0.00, 0.00, 1): 
    
    epsA = 0
    epsB = 0
    epsC = 0

    monolayer_C4 = make_system(type_C4="monolayer")
    
    #kwant.plot(monolayer_C4)
    
    monolayer_wrapped = kwant.wraparound.wraparound(monolayer_C4).finalized()   
    
    #vor, bz_vertices = First_BZ(monolayer_wrapped)

    par = {"|t$_A$|": str("{:.2f}".format(tA)), "|t$_B$|": str("{:.2f}".format(tB)), "|t$_C$|": str("{:.2f}".format(tC))}

    fig = plt.figure()
    ax = kwant.wraparound.plot_2d_bands(monolayer_wrapped, params = par, show = False)
    #ax.title(str(par), y = 1.3)
    ax.savefig('BS'+str("{:.2f}".format(x))+'.png')
    kwant.wraparound.plot_2d_bands(monolayer_wrapped)

    k_path_bandstructure(monolayer_wrapped, 'G', 'K', 'M','G')
    #k_path_bandstructure(bilayer_wrapped  , 'G', 'K', 'M','G')
    
                                                                                        
