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
L = 50 # size of the system (in each dimension)
averaging = 20 # number of runs for averaging DOS and conductivities

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
    
    for x in range(-int(L/2), int(L/2)):
        for y in range(-int(L/2), int(L/2)):

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
    return abs(x) < 500 and abs(y) < 500

def circle(site):
    x, y = site.pos
    r = 30
    return x ** 2 + y ** 2 < r ** 2

# Declare big things coming


for x in [0.00, 0.28, 0.33, 0.50, 1.00]:
    
    legend = []
    
    fig_dos, ax_dos = plt.subplots()
    fig_cond_xx, ax_cond_xx = plt.subplots()
    fig_cond_xy, ax_cond_xy = plt.subplots()
    e_ax_min = float("inf")
    e_ax_max = float("-inf")
    
    for d in [0.01, 0.1, 0.5, 0.8, 1.0, 1.5, 2.0]:

        all_energies = []
        all_densities = []
        all_cond_xx_miu = []
        all_cond_xy_miu = []
        all_T = []
        all_cond_xx_T = []
        all_cond_xy_T = []
        
        for av in range(0, averaging):
        
            # Hoppings
            
            t1 = -1.0 + x
            t2 = -x

            syst = kwant.Builder() 
            model = make_system(type_kagome = 'monolayer')
            area_per_site = np.abs(lat_const*lat_const*np.sqrt(3)/2)/3
            syst.fill(model, trunc, (50, 50));
            
            syst.eradicate_dangling()
            
            # Plot system before running
            
            #kwant.plot(syst);

            fsyst = syst.finalized()
        
            # Evaluate DOS
        
            rho = kwant.kpm.SpectralDensity(fsyst)
            energies, densities = rho()
            print("Averaging:", av+1,"/", averaging)
            
            # Evaluate conductivity tensor

            where = lambda s: np.linalg.norm(s.pos - np.array([0, 0])) < 1

            # xx component
        
            s_factory = kwant.kpm.LocalVectors(fsyst, where)
            cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x', num_vectors=None, vector_factory=s_factory)
            #cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x')
            
            cond_xx_miu = [cond_xx(mu = e, temperature = 0.01)/(area_per_site) for e in energies ]
            #print("-----mu------")
            #print([(cond_xx_miu[i], energies[i]) for i in range(0, len(energies))])
            #print()
            cond_xx_T = [cond_xx(mu = -1, temperature = T[i])/(area_per_site) for i in range(len(T)) ]

            # xy component

            s_factory = kwant.kpm.LocalVectors(fsyst, where)
            cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y', num_vectors=None, vector_factory=s_factory)
            #cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y')

            cond_xy_miu = [cond_xy(mu = e, temperature = 0.01)/(area_per_site) for e in energies ]
            cond_xy_T = [cond_xy(mu = -1, temperature = T[i])/(area_per_site) for i in range(len(T)) ] 
        
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
        
        min_est_e = est_energies[0]
        max_est_e = est_energies[-1]
        if (min_est_e < e_ax_min):
            e_ax_min = min_est_e
        if (max_est_e > e_ax_max):
            e_ax_max = max_est_e


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


        est_energies = [-10, min_est_e] + est_energies + [max_est_e, 10]
        est_densities = [0, 0] + est_densities + [0, 0]
        est_cond_xx_miu = [0, 0] + est_cond_xx_miu + [0, 0]
        est_cond_xy_miu = [0, 0] + est_cond_xy_miu + [0, 0]
        
        ############
        # Plotting #
        ############

        save_fig_to = '/mnt/d/Masters/KPM/DOS/Kagome/'
        legend.append("$\sigma$ = "+str("{:.2f}".format(d)))

        ax_dos.plot(est_energies, est_densities)
        ax_dos.set_title("Disordered Euler Kagome DOS with KPM (t = "+str(round(t1, 2))+", t' = "+str(round(t2, 2))+")", y=1.1)
        ax_dos.set_xlabel("Energy (E) / |t + t'|")
        ax_dos.set_ylabel(r"Density of states $\rho(E)$ (a.u.)")
                
        # Conductivity tensor xx component
        
        ax_cond_xx.plot(est_energies, est_cond_xx_miu)
        ax_cond_xx.set_title("Average conductivity $\sigma_{xx}$ at different chemical potentials \n (t = "+str(round(t1, 2))+", t' = "+str(round(t2, 2))+")", y=1.1)
        ax_cond_xx.set_xlabel(r"Chemical potential $\mu$")
        ax_cond_xx.set_ylabel(r"Conductivity $\langle \sigma_{xx} \rangle$ $(e^2/h)$")
        

        """ 
        plt.plot(T, est_cond_xx_T)
        plt.title("Average conductivity $\sigma_{xx}$ vs temperature for $\mu$ = 0 \n ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(t2, 2))+")", y=1.1)
        plt.xlabel(r"Temperature (T) eV$/k_B$")
        plt.ylabel(r"Conductivity $\langle \sigma_{xx} \rangle$ $(e^2/h)$")
        plt.xlim(T_min,T_max-0.01)

        plt.tight_layout()
        plt.savefig(save_fig_to + "Sigma_xx_T_dis_"+str(d)+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
        """
        # Conductivity tensor xy component

        ax_cond_xy.plot(est_energies, est_cond_xy_miu)
        ax_cond_xy.set_title("Average conductivity $\sigma_{xy}$ at different chemical potentials \n (t = "+str(round(t1, 2))+", t' = "+str(round(t2, 2))+")", y=1.1)
        ax_cond_xy.set_xlabel(r"Chemical potential $\mu$")
        ax_cond_xy.set_ylabel(r"Conductivity $\langle \sigma_{xy} \rangle$ $(e^2/h)$")
        
        """
        plt.plot(T, est_cond_xy_T)
        plt.title("Conductivity $\sigma_{xy}$ vs temperature for $\mu$ = 0 \n ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(t2, 2))+")", y=1.1)
        plt.xlabel(r"Temperature (T) eV$/k_B$")
        plt.ylabel(r"Conductivity $\langle \sigma_{xy} \rangle$ $(e^2/h)$")
        plt.xlim(T_min,T_max-0.01)

        plt.tight_layout()
        plt.savefig(save_fig_to + "Sigma_xy_T_dis_"+str(d)+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
        """

    ax_dos.set_xlim(e_ax_min, e_ax_max)
    ax_cond_xx.set_xlim(e_ax_min, e_ax_max)
    ax_cond_xy.set_xlim(e_ax_min, e_ax_max)
    ax_dos.set_ylim(0, 0.6)
    fig_dos.legend(legend, loc='upper left', bbox_to_anchor=(0.15, 0.8))
    fig_cond_xx.legend(legend, loc='upper right', bbox_to_anchor=(0.90, 0.8))
    fig_cond_xy.legend(legend, loc='upper right', bbox_to_anchor=(0.90, 0.8))
    fig_dos.tight_layout() 
    fig_cond_xx.tight_layout() 
    fig_cond_xy.tight_layout() 
    fig_dos.savefig(save_fig_to + "DOS_dis_full_tNNN_"+str(x)+".png", dpi='figure')
    fig_cond_xx.savefig(save_fig_to + "Sigma_xx_mu_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
    fig_cond_xy.savefig(save_fig_to + "Sigma_xy_mu_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')

