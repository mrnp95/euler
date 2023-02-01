import kwant 
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.linalg as la
from numpy import sqrt
import tinyarray as ta
import matplotlib.pyplot as plt


# Default settings

lat_const = 1  # lattice constant of graphene (unit: nm)
t1 = 1.0    # nearest neighbor hopping parameter for graphene (unit: eV)
t2 = -0.5*(1j)    # next nearest neighbor hopping parameter for graphene (unit: eV)
d  = 1.0    # standard deviation in Gaussian disorder (unit: eV)
L = 200 # size of the system (in each dimension)
averaging = 200 # number of runs for averaging DOS and conductivities

# Domains of cond function for later

N_bins = 200 # Bins for energies in the estimator
N_binsT = 500 # Bins for temperature
T_min = 0.01 
T_max = 5.00
T = np.linspace(T_min, T_max, N_binsT)

def mod(a,b):
    mod = ((a % b) + b) % b
    return mod

def make_system(type_graphene = 'monolayer'):
    Bravais_vector = [(    lat_const,                     0), 
                      (0.5*lat_const, 0.5*lat_const*sqrt(3))] # Bravais vectors
    Bottom_Lat_pos = [(0, lat_const/sqrt(3)), (0,  0)]    # The position of sublattice atoms in the layer  
    
    Bottom_lat = kwant.lattice.general(Bravais_vector, Bottom_Lat_pos, norbs=1)
    B_sub_a, B_sub_b = Bottom_lat.sublattices
    
    sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder()
    
    for x in range(0, int(L)):
        for y in range(0, int(L)):

	    # define hopping and on-site potential on bottom layer graphene

            bulk[B_sub_a(x,y)] = np.random.normal(0, d, 1)
            bulk[B_sub_b(x,y)] = np.random.normal(0, d, 1)
    
    for x in range(0, int(L)):
        for y in range(0, int(L)):
     
            bulk[B_sub_a(x,y),B_sub_b(x,y)] = t1
            bulk[B_sub_a(x,y),B_sub_b(x,mod(y+1,L))] = t1
            bulk[B_sub_a(x,y),B_sub_b(mod(x-1,L),mod(y+1,L))] = t1
    
            # Next neighbors

            bulk[B_sub_b(x,y),B_sub_b(mod(x+1,L),y)] = -t2
            bulk[B_sub_b(x,y),B_sub_b(x,mod(y+1,L))] = -t2
            bulk[B_sub_b(x,y),B_sub_b(mod(x+1,L), mod(y-1,L))] = -t2 
            bulk[B_sub_a(x,y),B_sub_a(mod(x+1,L),y)] = t2
            bulk[B_sub_a(x,y),B_sub_a(x,mod(y+1,L))] = t2
            bulk[B_sub_a(x,y),B_sub_a(mod(x+1,L),mod(y-1,L))] = t2
    
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


for x in [0.00, 0.50]:
    
    legend = []
    
    for length in [160, 180, 200]:
        
        L = length
    
        for d in [0.00, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.90, 1.0, 1.2, 1.5]:
            
            print("|t'| = "+str(x)+", d = "+str(d))

            all_energies = []
            all_densities = []
            all_cond_xx_miu = []
            all_cond_xy_miu = []
            #all_T = []
            #all_cond_xx_T = []
            #all_cond_xy_T = []

            for av in range(0, averaging):
            
                # Hoppings
                
                t1 = 1.0
                t2 = x*(1j)

                syst = kwant.Builder() 
                model = make_system(type_graphene = 'monolayer')
                #kwant.plot(model)
                #model = make_syst_topo()
                area_per_site = np.abs(lat_const*lat_const*np.sqrt(3)/2)/2
                syst.fill(model, trunc, (0,0));
                     
                # Debugging...

                #syst[model.shape(trunc,(0,0))] = 0.
                #syst[model.neighbors()] = t1
                # add second neighbours hoppings
                #syst[model.a.neighbors()] = -t2
                #syst[model.b.neighbors()] = t2
                
                syst.eradicate_dangling()
                
                # Plot system before running
                
                #kwant.plot(syst);

                kwant.wraparound.wraparound(syst)
                fsyst = syst.finalized()
            
                # Evaluate DOS
            
                rho = kwant.kpm.SpectralDensity(fsyst, num_moments = 4096, num_vectors = 1000)

                energies, densities = rho()
                #print(len(energies), len(densities))
                print("Averaging:", av+1,"/", averaging)
                
                           # Evaluate conductivity tensor

                where = lambda s: np.linalg.norm(s.pos) < 1

                # xx component
            
                s_factory = kwant.kpm.LocalVectors(fsyst, where)
                cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x', num_vectors=None, vector_factory=s_factory)
                #cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x')
                
                cond_xx_miu = [cond_xx(mu = e, temperature = 0)/(area_per_site) for e in energies ]
                #print("-----mu------")
                #print([(cond_xx_miu[i], energies[i]) for i in range(0, len(energies))])
                #print()
                #cond_xx_T = [cond_xx(mu = 0, temperature = T[i])/(area_per_site) for i in range(len(T)) ]

                # xy component

                s_factory = kwant.kpm.LocalVectors(fsyst, where)
                cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y', num_vectors=None, vector_factory=s_factory)
                #cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y')

                cond_xy_miu = [cond_xy(mu = e, temperature = 0)/(area_per_site) for e in energies ]
                #cond_xy_T = [cond_xy(mu = 0, temperature = T[i])/(area_per_site) for i in range(len(T)) ] 
            
                # For estimator
                
                all_energies += list(energies)
                all_densities += list(densities)
                all_cond_xx_miu += list(cond_xx_miu)
                all_cond_xy_miu += list(cond_xy_miu)
                #all_T += list(T)
                #all_cond_xx_T += list(cond_xx_T)
                #all_cond_xy_T += list(cond_xy_T)

            #############
            # Estimator #
            #############
            
            E_min = min(all_energies)
            E_max = max(all_energies)

            est_energies = [0] * N_bins
            est_densities = [0] * N_bins
            est_cond_xx_miu = [0] * N_bins
            est_cond_xy_miu = [0] * N_bins
            #est_cond_xx_T = [0] * N_binsT
            #est_cond_xy_T = [0] * N_binsT
            bin_n = [0] * N_bins
            #bin_nT = [0] * N_binsT
            
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
                    return 0.000001
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
            """

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
            """
    
            ############
            # SAVING   #
            ############

            file_name = "Haldane_DOS_dis_full_"+str(d)+"_tNNN_"+str(x)+"_L_"+str(L)+".dat"
            with open(file_name, "x") as file_fd:
                for i in range(0, len(est_energies)):
                    file_fd.write(str(est_energies[i])+" "+str(est_densities[i])+" "+str(est_cond_xx_miu[i])+" "+str(est_cond_xy_miu[i])+"\n")

    """       
            ############
            # Plotting #
            ############

            save_fig_to = './'

            plt.plot(est_energies, est_densities)
            plt.title("Disordered Haldane DOS with KPM ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(np.abs(t2),2))+")", y=1.1)
            plt.xlim(E_min,E_max)
            #plt.ylim(0, 500)
            plt.xlabel("Energy (E) / t")
            plt.ylabel(r"Density of states $\rho(E)$ (a.u.)")
            plt.tight_layout()
            #plt.margins(x = 1.5, y = 1.0)
            #plt.show()
        
            plt.savefig(save_fig_to + "DOS_dis_full_"+str(d)+"_tNNN_"+str(x)+".png", dpi='figure')
            plt.clf()

            # Conductivity tensor xx component

            plt.plot(est_energies, est_cond_xx_miu)
            plt.title("Zero temperature conductivity $\sigma_{xx}$ at different chemical potentials \n ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(np.abs(t2),2))+")", y=1.1)
            plt.xlabel(r"Chemical potential $\mu$")
            plt.ylabel(r"Conductivity $\sigma_{xx}$ / $e^2/h$")
            plt.xlim(E_min, E_max)
            plt.tight_layout()
            plt.savefig(save_fig_to + "Sigma_xx_mu_dis_"+str(d)+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
            plt.clf()
            
            plt.plot(T, est_cond_xx_T)
            plt.title("Conductivity $\sigma_{xx}$ vs temperature for $\mu$ = 0 \n ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(np.abs(t2),2))+")", y=1.1)
            plt.xlabel(r"Temperature (T) eV$/k_B$")
            plt.ylabel(r"Conductivity $\sigma_{xx}$ / $e^2/h$")
            plt.xlim(T_min,T_max)

            plt.tight_layout()
            plt.savefig(save_fig_to + "Sigma_xx_T_dis_"+str(d)+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
            plt.clf()
            
            # Conductivity tensor xy component

            plt.plot(est_energies, est_cond_xy_miu)
            plt.title("Zero temperature conductivity $\sigma_{xy}$ at different chemical potentials \n ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(np.abs(t2),2))+")", y=1.1)
            plt.xlabel(r"Chemical potential $\mu$")
            plt.ylabel(r"Conductivity $\sigma_{xy}$ / $e^2/h$")
            plt.xlim(E_min, E_max)
            plt.tight_layout()
            plt.savefig(save_fig_to + "Sigma_xy_mu_dis_"+str(d)+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
            plt.clf()
            
            plt.plot(T, est_cond_xy_T)
            plt.title("Conductivity $\sigma_{xy}$ vs temperature for $\mu$ = 0 \n ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(np.abs(t2),2))+")", y=1.1)
            plt.xlabel(r"Temperature (T) eV$/k_B$")
            plt.ylabel(r"Conductivity $\sigma_{xy}$ / $e^2/h$")
            plt.xlim(T_min,T_max)

            plt.tight_layout()
            plt.savefig(save_fig_to + "Sigma_xy_T_dis_"+str(d)+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
            plt.clf()
    """        
