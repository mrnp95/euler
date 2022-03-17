import kwant 
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.linalg as la
from numpy import sqrt
import tinyarray as ta
import matplotlib.pyplot as plt
import functools as ft
import mirror_chern_original as ch
from kpm_funcs import position_operator

# Default settings

lat_const = 1  # lattice constant of square lattice (unit: nm)
t0 = 1.00    # hoppings (unit: eV)
t1 = 1.00
m = 1.00
delta = 0.50

epsA = 0.5
epsB = -0.5
epsC = -0.5
epsD = 0.5


d  = 1.0    # standard deviation in Gaussian disorder (unit: eV)
L = 100 # size of the system (in each dimension)
averaging = 10 # number of runs for averaging DOS and conductivities
disorders = [0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
m_s = [1.00, 0.00]

# Domains of cond function for later

N_bins = 100 # Bins for energies in the estimator
N_binsT = 500 # Bins for temperature
T_min = 0.01 
T_max = 5.00
T = np.linspace(T_min, T_max, N_binsT)


##########
# BUILD  #
##########

def make_system(type_4band = 'monolayer'):
    
    Bravais_vector = [(lat_const, 0, 0), 
                        (0,  lat_const, 0),
			(0 , 0,  20)] # Bravais vectors
    Bottom_Lat_pos = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]    # The position of sublattice atoms in Bottom layer  
    Bottom_lat = kwant.lattice.general(Bravais_vector, Bottom_Lat_pos, norbs=1)
    B_sub_A, B_sub_B, B_sub_C, B_sub_D = Bottom_lat.sublattices 
    
    sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder(sym)

    # define hoppings and on-site potentials
    
    bulk[B_sub_A(0,0,0)] = epsA*0+delta+np.random.normal(0, d, 1)
    bulk[B_sub_B(0,0,0)] = epsB*0-delta+np.random.normal(0, d, 1)
    bulk[B_sub_C(0,0,0)] = epsC*0-delta+np.random.normal(0, d, 1)
    bulk[B_sub_D(0,0,0)] = epsD*0+delta+np.random.normal(0, d, 1)
    
    bulk[kwant.builder.HoppingKind((0,1,0), B_sub_A,B_sub_A)] = t0
    bulk[kwant.builder.HoppingKind((0,-1,0), B_sub_A,B_sub_A)] = -t0
    bulk[kwant.builder.HoppingKind((0,1,0), B_sub_C,B_sub_C)] = t0
    bulk[kwant.builder.HoppingKind((0,-1,0), B_sub_C,B_sub_C)] = -t0
    
    bulk[kwant.builder.HoppingKind((0,1,0), B_sub_B,B_sub_B)] = -t0
    bulk[kwant.builder.HoppingKind((0,-1,0), B_sub_B,B_sub_B)] = t0
    bulk[kwant.builder.HoppingKind((0,1,0), B_sub_D,B_sub_D)] = -t0
    bulk[kwant.builder.HoppingKind((0,-1,0), B_sub_D,B_sub_D)] = t0 
   
    bulk[kwant.builder.HoppingKind((1,0,0), B_sub_A,B_sub_B)] = t0
    bulk[kwant.builder.HoppingKind((-1,0,0), B_sub_A,B_sub_B)] = -t0
    
    bulk[kwant.builder.HoppingKind((1,0,0), B_sub_C,B_sub_D)] = t0
    bulk[kwant.builder.HoppingKind((-1,0,0), B_sub_C,B_sub_D)] = -t0
  
    bulk[kwant.builder.HoppingKind((0,0,0), B_sub_A,B_sub_D)] = m
    bulk[kwant.builder.HoppingKind((0,0,0), B_sub_B,B_sub_C)] = -m

    bulk[kwant.builder.HoppingKind((0,1,0), B_sub_A,B_sub_D)] = -t1
    bulk[kwant.builder.HoppingKind((0,-1,0), B_sub_A,B_sub_D)] = -t1
    bulk[kwant.builder.HoppingKind((1,0,0), B_sub_A,B_sub_D)] = -t1
    bulk[kwant.builder.HoppingKind((-1,0,0), B_sub_A,B_sub_D)] = -t1
     
    bulk[kwant.builder.HoppingKind((0,1,0), B_sub_B,B_sub_C)] = t1
    bulk[kwant.builder.HoppingKind((0,-1,0), B_sub_B,B_sub_C)] = t1
    bulk[kwant.builder.HoppingKind((1,0,0), B_sub_B,B_sub_C)] = t1
    bulk[kwant.builder.HoppingKind((-1,0,0), B_sub_B,B_sub_C)] = t1   

    return bulk

# Different geometries of the finite system

def trunc(site):
    x, y, z = abs(site.pos)
    return abs(x) <= 50 and abs(y) <= 50

def circle(site):
    x, y = site.pos
    r = 30
    return x ** 2 + y ** 2 < r ** 2

# Declare big things coming

for t in m_s:
    
    legend = []
    
    C_final = []

    C_final_std = []

    fig_C, ax_C = plt.subplots()
    
    for d in disorders:
        
        print("m = "+str(t)+", d = "+str(d))

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
            
            epsA = t
            epsB = t
            epsC = t

            syst = kwant.Builder() 
            model = make_system(type_4band = 'monolayer')
            #kwant.plot(model)
            #model = make_syst_topo()
            area_per_site = np.abs(lat_const*lat_const)
            syst.fill(model, trunc, (0, 0, 0));
             
            syst.eradicate_dangling()
            
            # Plot system before running
            
            #kwant.plot(syst);


            fsyst = syst.finalized()

            # Chern markers

            num_vectors = 2
            num_moments = 1000
            A = area_per_site*L*L

            C_list = []

            systw = kwant.wraparound.wraparound(syst)   
            fsystw = systw.finalized()
            x, y, z = position_operator(fsystw)
            
            """
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
            
            #window1 = ch.make_window(fsystw, shape1)
            #window2 = ch.make_window(fsystw, shape2)
            #windows = [window1, window2]
            windows = [ch.make_window(fsystw, trunc)]

            #vectors = np.array(np.vstack(np.array(np.vstack(np.array([i+0.5*j,0.5*j*np.sqrt(3)]) for i in range(int(-L/2), int(L/2)))) for j in range(int(-L/2), int(L/2)))) 
            #vectors2 = np.array(np.vstack(np.array(np.vstack(np.array([i+0.5*j, 0.5*j*np.sqrt(3)+1/np.sqrt(3)]) for i in range(int(-L/2),int(L/2)))) for j in range(int(-L/2),int(L/2))))
            vectors = np.array(np.vstack(np.array(np.vstack(np.array([i,j]) for i in range(int(-L/2), int(L/2)))) for j in range(int(-L/2), int(L/2)))) 
            vectors2 = np.array(np.vstack(np.array(np.vstack(np.array([i,j+1/np.sqrt(3)]) for i in range(int(-L/2),int(L/2)))) for j in range(int(-L/2),int(L/2))))

            
            print(vectors.shape, vectors2.shape)
            vectors = np.array(np.vstack((vectors,vectors2))).T
            print(vectors.shape)
            """
            #tags = np.array([[i,j] for i in range(int(-L*0/2), int(L/2)) for j in range(int(-L/2), int(L/2))])
            #tags = np.array([[i,j] for i in [0, -15] for j in [0, 15]])
            tags = np.array([np.hstack([np.random.randint(-int(L/2), high=(int(L/2)), size=2), 0]) for i in range(0, num_vectors)])
            vectors = []
            Cs = 0
            C_all = []
            err = 0
            
            n = np.array([1, 0, 0])
            M_trf = ft.partial(ch.M_cubic, n=n)
            UM = ch.UM_s(n)
            M = ch.pg_op(fsystw, M_trf, UM)
            print(M)
            
            for tag in tags:

                where = lambda s: s.tag == tag
                vector_factory = kwant.kpm.LocalVectors(fsyst, where)
                vectors = vector_factory
                #C_list = [ch.mirror_chern(fsyst, x, y, Mz=None, vectors=num_vectors, e_F=0, kpm_params=dict(num_moments=num_moments), params=None, bounds=None, window=window, return_std=False) for window in windows]
            
                C_list = ch.mirror_chern(fsyst, x, y, Mz=M, vectors=vectors, e_F=0, kpm_params=dict(num_moments=num_moments), params=None, bounds=None, window=None, return_std=False)
                C_list = np.array(C_list)
                Cs = np.sum(C_list, axis = 0) / (area_per_site*2)
                print(Cs, tag)
                if np.abs(Cs) > 2: 
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
    ax_C.set_title("Averaged local mirror Chern number C$_M$ as a function of disorder \n (mean = "+str(round(t, 2))+")", y=1.1)
    ax_C.set_xlabel(r"Disorder strength $\sigma$")
    ax_C.set_ylabel(r"Local mirror Chern number averaged over system")
    ax_C.set_xlim(int(disorders[0]), int(disorders[len(disorders)-1]))
    #fig_C.legend(legend, loc='upper right', bbox_to_anchor=(0.90, 0.8))       
    fig_C.tight_layout()
    save_fig_to = './'
    fig_C.savefig(save_fig_to + "Mirror_marking_4band11_mean_"+str(t)+".png", bbox_inches = 'tight', dpi='figure')

    ############
    # SAVING   #
    ############

    file_name = "4band11_Chern_mirror_marking_dis_full_"+"_mean_"+str(t)+"_L_"+str(L)+".dat"
    with open(file_name, "x") as file_fd:
        for i in range(0, len(C_final)):
            file_fd.write(str(disorders[i])+" "+str(C_final[i])+" "+str(C_final_std[i])+"\n")

