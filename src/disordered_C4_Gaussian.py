import kwant 
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.linalg as la
from numpy import sqrt
import tinyarray as ta
import matplotlib.pyplot as plt


# Default settings

lat_const = 1  # lattice constant of square lattice (unit: nm)
tA = 0.35    # hoppings (unit: eV)
tB = 0.46
tC = 0.69    
epsA = 0
epsB = 0
epsC = 0

d  = 1.0    # standard deviation in Gaussian disorder (unit: eV)
L = 100 # size of the system (in each dimension)
averaging = 200 # number of runs for averaging DOS and conductivities

# Domains of cond function for later

N_bins = 100 # Bins for energies in the estimator
N_binsT = 500 # Bins for temperature
T_min = 0.01 
T_max = 5.00
T = np.linspace(T_min, T_max, N_binsT)


##########
# BUILD  #
##########

def make_system(type_C4 = 'monolayer'):
    
    Bravais_vector = [(lat_const, 0), 
                        (0,  lat_const)] # Bravais vectors
    Bottom_Lat_pos = [(0, 0), (0, 0), (0, 0)]    # The position of sublattice atoms in Bottom layer  
    Bottom_lat = kwant.lattice.general(Bravais_vector, Bottom_Lat_pos, norbs=1)
    B_sub_a, B_sub_b, B_sub_c = Bottom_lat.sublattices 
    
    sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder()

    # define hoppings and on-site potentials
    
    for x in range(-int(L/2), int(L/2)):
        for y in range(-int(L/2), int(L/2)):
            
            # define hopping and on-site potential on bottom layer kagome

            bulk[B_sub_a(x,y)] = epsA+np.random.normal(0, d, 1)
            bulk[B_sub_b(x,y)] = epsB+np.random.normal(0, d, 1)
            bulk[B_sub_c(x,y)] = epsC+np.random.normal(0, d, 1)
    
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

# Different geometries of the finite system

def trunc(site):
    x, y = abs(site.pos)
    return abs(x) < 500 and abs(y) < 500

def circle(site):
    x, y = site.pos
    r = 30
    return x ** 2 + y ** 2 < r ** 2

# Declare big things coming


for x in [0.00]:
    
    legend = []
    
    for d in [0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]:
        
        print("mean = "+str(x)+", d = "+str(d))

        all_energies = []
        all_densities = []
        all_cond_xx_miu = []
        all_cond_xy_miu = []
        all_T = []
        all_cond_xx_T = []
        all_cond_xy_T = []
        
        for av in range(0, averaging):
        
            # Hoppings
            
            epsA = x
            epsB = x
            epsC = x

            syst = kwant.Builder() 
            model = make_system(type_C4 = 'monolayer')
            area_per_site = np.abs(lat_const*lat_const)
            syst.fill(model, trunc, (0, 0));
            
            syst.eradicate_dangling()
            
            # Plot system before running
            
            #kwant.plot(syst);

            fsyst = syst.finalized()
        
            # Evaluate DOS
        
            rho = kwant.kpm.SpectralDensity(fsyst)
            energies, densities = rho()
            print("Averaging:", av+1,"/", averaging)
            
            # Evaluate conductivity tensor

            where = lambda s: np.linalg.norm(s.pos) < 1

            # xx component
        
            s_factory = kwant.kpm.LocalVectors(fsyst, where)
            cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x', num_vectors=None, vector_factory=s_factory)
            #cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x')
            
            cond_xx_miu = [cond_xx(mu = e, temperature = 0.01)/(area_per_site) for e in energies ]
            #print("-----mu------")
            #print([(cond_xx_miu[i], energies[i]) for i in range(0, len(energies))])
            #print()
            cond_xx_T = [cond_xx(mu = 0, temperature = T[i])/(area_per_site) for i in range(len(T)) ]

            # xy component

            s_factory = kwant.kpm.LocalVectors(fsyst, where)
            cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y', num_vectors=None, vector_factory=s_factory)
            #cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y')

            cond_xy_miu = [cond_xy(mu = e, temperature = 0.01)/(area_per_site) for e in energies ]
            cond_xy_T = [cond_xy(mu = 0, temperature = T[i])/(area_per_site) for i in range(len(T)) ] 
        
            # For estimator
            
            all_energies += list(energies)
            all_densities += list(densities)
            all_cond_xx_miu += list(cond_xx_miu)
            all_cond_xy_miu += list(cond_xy_miu)
            all_T += list(T)
            all_cond_xx_T += list(cond_xx_T)
            all_cond_xy_T += list(cond_xy_T)

        #############
        # Estimator #
        #############
        
        E_min = min(all_energies)
        E_max = max(all_energies)

        est_energies = [0] * N_bins
        est_densities = [0] * N_bins
        est_cond_xx_miu = [0] * N_bins
        est_cond_xy_miu = [0] * N_bins
        est_cond_xx_T = [0] * N_binsT
        est_cond_xy_T = [0] * N_binsT
        bin_n = [0] * N_bins
        bin_nT = [0] * N_binsT
        
        # Conductivity as function of chemical potential, setting up DOS for averaging

        for i in range(0, len(all_energies)):
            bin = int(np.floor((all_energies[i]-E_min)/(E_max-E_min) * N_bins))
            if (bin >= N_bins or bin < 0):
                continue
            
            est_energies[bin] += all_energies[i]
            est_densities[bin] += all_densities[i]
            est_cond_xx_miu[bin] += all_cond_xx_miu[i]
            est_cond_xy_miu[bin] += all_cond_xy_miu[i]
            
            bin_n[bin] += 1

        def not_zero(b):
            if b == 0:
                return 1
            return b
        bin_n = [not_zero(b) for b in bin_n]


        est_energies = [est_energies[i]/(bin_n[i]) for i in range(0, N_bins)]
        est_densities = [est_densities[i]/(bin_n[i]) for i in range(0, N_bins)]
        est_cond_xx_miu = [est_cond_xx_miu[i]/(bin_n[i]) for i in range(0, N_bins)]
        est_cond_xy_miu = [est_cond_xy_miu[i]/(bin_n[i]) for i in range(0, N_bins)]
        
        # Normalisation of DOS

        Norm_D = 0

        for i in range(0, len(est_energies)):
            Norm_D += est_densities[i]*(E_max-E_min)/N_bins
         
        est_densities = [est_densities[i]/Norm_D for i in range(0, N_bins)]

        # Conductivity as function of temperature


        for i in range(0, len(all_T)):
            bin = int(np.floor((all_T[i]-T_min)/(T_max-T_min) * N_binsT))
            if (bin >= N_binsT or bin < 0):
                continue
        
            est_cond_xx_T[bin] += all_cond_xx_T[i]
            est_cond_xy_T[bin] += all_cond_xy_T[i]

            bin_nT[bin] += 1

        def not_zero(b):
            if b == 0:
                return 1
            return b
        bin_nT = [not_zero(b) for b in bin_nT]
 

        est_cond_xx_T = [est_cond_xx_T[i]/(bin_nT[i]) for i in range(0, N_binsT)]
        est_cond_xy_T = [est_cond_xy_T[i]/(bin_nT[i]) for i in range(0, N_binsT)]


        ############
        # SAVING   #
        ############

        file_name = "Square_DOS_dis_full_"+str(d)+"_mean_"+str(x)+"_L_"+str(L)+".dat"
        with open(file_name, "x") as file_fd:
            for i in range(0, len(est_energies)):
                file_fd.write(str(est_energies[i])+" "+str(est_densities[i])+" "+str(est_cond_xx_miu[i])+" "+str(est_cond_xy_miu[i])+"\n")

        file_name = "Square_T_dis_full_"+str(d)+"_mean_"+str(x)+"_L_"+str(L)+".dat"
        with open(file_name, "x") as file_fd:
            for i in range(0, N_binsT):
                file_fd.write(str(T[i])+" "+str(est_cond_xx_T[i])+" "+str(est_cond_xy_T[i])+"\n")
