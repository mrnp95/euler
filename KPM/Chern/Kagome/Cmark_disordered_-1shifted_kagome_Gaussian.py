import kwant 
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.linalg as la
from numpy import sqrt
import tinyarray as ta
import matplotlib.pyplot as plt
from kpm_funcs import position_operator
import mirror_chern as ch
import scipy.sparse 
import functools as ft

# Default settings

lat_const = 1  # lattice constant of kagome (unit: nm)
t1 = -1.0    # nearest neighbor hopping parameter for kagome (unit: eV)
t2 = -1.0*0.0    # next nearest neighbor hopping parameter for kagome (unit: eV)
tn = 0.0    # interlayer hopping between kagomes (unit: eV)
d  = 1.0    # standard deviation in Gaussian disorder (unit: eV)
L = 100 # size of the system (in each dimension)
averaging = 5 # number of runs for averaging DOS and conductivities

miu = -1
disorders = [0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
hoppings = [0.00, 0.28, 0.33, 0.50, 1.00]

# Domains of cond function for later

N_bins = 100 # Bins for energies in the estimator
N_binsT = 500 # Bins for temperature
T_min = 0.01 
T_max = 5.00
T = np.linspace(T_min, T_max, N_binsT)

##########
# BUILD  #
##########

def make_system(type_kagome = 'monolayer'):
    
    Bravais_vector = [(    lat_const,                     0, 0), 
                      (0.5*lat_const, 0.5*lat_const*sqrt(3), 0),
                      (0, 0, 20)] # Bravais vectors
    Bottom_Lat_pos = [(0.5*lat_const, 0, 0), (0.25*lat_const, 0.25*lat_const*sqrt(3), 0), (0,  0, 0)]    # The position of sublattice atoms in Bottom layer  
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

            bulk[B_sub_a(x,y,0)] = -1+np.random.normal(0, d, 1)
            bulk[B_sub_b(x,y,0)] = -1+np.random.normal(0, d, 1)
            bulk[B_sub_c(x,y,0)] = -1+np.random.normal(0, d, 1)
     
    bulk[kwant.builder.HoppingKind((0,0,0), B_sub_b,B_sub_a)] = t1
    bulk[kwant.builder.HoppingKind((0,0,0), B_sub_c,B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((0,0,0), B_sub_a,B_sub_c)] = t1
    bulk[kwant.builder.HoppingKind((-1,0,0), B_sub_a,B_sub_c)] = t1
    bulk[kwant.builder.HoppingKind((0,1,0), B_sub_c,B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((1,-1,0),B_sub_b,B_sub_a)] = t1
    
        # Next neighbors

    bulk[kwant.builder.HoppingKind((0,1,0), B_sub_c,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((1,-1,0), B_sub_c,B_sub_a)] = t2
    #bulk[kwant.builder.HoppingKind((1,0), B_sub_b,B_sub_a)] = t2
    #bulk[kwant.builder.HoppingKind((0,-1), B_sub_b,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((-1,0,0), B_sub_a,B_sub_b)] = t2
    bulk[kwant.builder.HoppingKind((0,1,0),B_sub_a,B_sub_b)] = t2
    #bulk[kwant.builder.HoppingKind((1,0), B_sub_c,B_sub_b)] = t2
    #bulk[kwant.builder.HoppingKind((-1,1), B_sub_c,B_sub_b)] = t2
    bulk[kwant.builder.HoppingKind((1,-1,0), B_sub_b,B_sub_c)] = t2
    bulk[kwant.builder.HoppingKind((-1,0,0), B_sub_b,B_sub_c)] = t2
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
    x, y, z = abs(site.pos)
    return abs(x) < 800 and abs(y) < 800

def circle(site):
    x, y = site.pos
    r = 30
    return x ** 2 + y ** 2 < r ** 2

# Declare big things coming


for t in hoppings:
    
    legend = []
    
    C_final = []

    C_final_std = []

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
        C_av_std = 0

        for av in range(0, averaging):
        
            # Hoppings
            
            t1 = -1.0 + t
            t2 = -t

            syst = kwant.Builder() 
            model = make_system(type_kagome = 'monolayer')
            area_per_site = np.abs(lat_const*lat_const*np.sqrt(3)/2)/3
            syst.fill(model, trunc, (0, 0, 0));
            
            syst.eradicate_dangling()
            
            # Plot system before running
            
            #kwant.plot(syst);


            fsyst = syst.finalized()

            # Chern markers

            num_vectors = 5
            num_moments = 1000
            A = area_per_site*3*L*L

            C_list = []

            systw = kwant.wraparound.wraparound(syst)   
            fsystw = systw.finalized()
            x, y, z = position_operator(fsystw)
            
            """
            windows = [ch.make_window(fsystw, trunc)]

            vectors = np.array(np.vstack(np.array(np.vstack(np.array(np.array([i/2,j])) for i in range(int(-L), int(L)))) for j in range(int(-L/2), int(L/2))))
            vectors2 = np.array(np.vstack(np.array(np.vstack(np.array([i, (2*j+1)/2]) for i in range(int(-L/2),int(L/2)))) for j in range(int(-L/2),int(L/2))))
            print(vectors.shape, vectors2.shape)
            vectors = np.array(np.vstack((vectors,vectors2))).T
            print(vectors.size)
            #vectors = np.array(np.vstack(np.array([i,0]) for i in range(int(-L/2), int(L/2))))
            #print(vectors.shape)
            #print(vectors)
			
            """
            #tags = np.array([[i,j] for i in range(int(-L*0/2), int(L/2)) for j in range(int(-L/2), int(L/2))])
            #tags = np.array([[i,j] for i in [0, -15] for j in [0, 15]])
            tags = np.array([np.hstack([np.random.randint(-int(L/2)+15, high=(int(L/2)-15), size=2), 0]) for i in range(0, num_vectors)])
            vectors = []
            Cs = 0
            C_all = []
            err = 0
            
            """
            n = np.array([0, 0, 1])
            M_trf = ft.partial(ch.M_cubic, n=n)
            UM = ch.UM_s(n)
            M = ch.pg_op(fsystw, M_trf, UM)
            #print(M)
            """
            for tag in tags:

                where = lambda s: s.tag == tag
                vector_factory = kwant.kpm.LocalVectors(fsyst, where)
                vectors = vector_factory
                #C_list = [ch.mirror_chern(fsyst, x, y, Mz=None, vectors=num_vectors, e_F=0, kpm_params=dict(num_moments=num_moments), params=None, bounds=None, window=window, return_std=False) for window in windows]
            
                C_list = ch.mirror_chern(fsyst, x, y, Mz=None, vectors=vectors, e_F=miu, kpm_params=dict(num_moments=num_moments), params=None, bounds=None, window=None, return_std=False)
                C_list = np.array(C_list)
                Cs = np.sum(C_list, axis = 0) / (area_per_site*3)
                print(Cs, tag)
                if np.abs(Cs) > 10: 
                    err += err
                    print('Numerical error!')
                else:
                    C_all.append(Cs)

            #Cs = np.sum(C_list, axis = 0) / vectors.shape[1]
            #Cs = np.sum(C_list, axis = 0)

            C = np.mean(np.array(C_all))
            C_std = np.std(np.array(C_all))
            print('Sampled Local Chern = '+str(C)+', Stdev of Local Chern = '+str(C_std))
            C_av += 1/(averaging - err)*C
            C_av_std += 1/(averaging - err)*C_std

        print('Averaged Local Chern = '+str(C_av))
        C_final.append(C_av)
        C_final_std.append(C_av_std)

    ax_C.plot(disorders, C_final)
    ax_C.set_title("Averaged local Chern number C as a function of disorder \n (|t| = "+str(round(np.abs(t1), 2))+", |t'| = "+str(round(np.abs(t2), 2))+")", y=1.1)
    ax_C.set_xlabel(r"Disorder strength $\sigma$")
    ax_C.set_ylabel(r"Local Chern number averaged over system")
    ax_C.set_xlim(int(disorders[0]), int(disorders[len(disorders)-1]))
    #fig_C.legend(legend, loc='upper right', bbox_to_anchor=(0.90, 0.8))       
    fig_C.tight_layout()
    save_fig_to = './'
    fig_C.savefig(save_fig_to + "Marking_-1shifted_Euler_tNNN_"+str(t)+".png", bbox_inches = 'tight', dpi='figure')

    ############
    # SAVING   #
    ############

    file_name = "Euler_Chern_-1shifted_marking_dis_full_"+"_tNNN_"+str(t)+"_L_"+str(L)+".dat"
    with open(file_name, "x") as file_fd:
        for i in range(0, len(C_final)):
            file_fd.write(str(disorders[i])+" "+str(C_final[i])+" "+str(C_final_std[i])+"\n")
