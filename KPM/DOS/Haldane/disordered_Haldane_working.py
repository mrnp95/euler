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
tn = 0.0    # interlayer hopping between graphene (unit: eV)
d  = 1.0    # standard deviation in Gaussian disorder (unit: eV)
L = 200 # size of the system (in each dimension)
averaging = 1 # number of runs for averaging DOS and conductivities

# Domains of cond function for later

miu = np.linspace(-5, 5, 1000)
T = np.linspace(0, 5, 300)

def make_system(type_graphene = 'monolayer'):
    Bravais_vector = [(    lat_const,                     0), 
                      (0.5*lat_const, 0.5*lat_const*sqrt(3))] # Bravais vectors
    Bottom_Lat_pos = [(0, lat_const/sqrt(3)), (0,  0)]    # The position of sublattice atoms in the layer  
    
    Bottom_lat = kwant.lattice.general(Bravais_vector, Bottom_Lat_pos, norbs=1)
    B_sub_a, B_sub_b = Bottom_lat.sublattices  
    
    #sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder()
    
    for x in range(int(-L/2), int(L/2)):
        for y in range(int(-L/2), int(L/2)):

	    # define hopping and on-site potential on bottom layer kagome

            bulk[B_sub_a(x,y)] = np.random.normal(0, d, 1)
            bulk[B_sub_b(x,y)] = np.random.normal(0, d, 1)
    
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

def make_syst_topo(a=1, t=1, t2=0.5):
    lat = kwant.lattice.honeycomb(a, norbs=1, name=['a', 'b'])
    return lat

def trunc(site):
    x, y = abs(site.pos)
    return abs(x) < 500 and abs(y) < 500

def circle(site):
    x, y = site.pos
    r = 30
    return x ** 2 + y ** 2 < r ** 2


for d in [0, 0.5, 1.0, 1.5, 2.0]:
    for x in [1.00]:
        for av in range(0, averaging):
        
            t1 = 1.0
            t2 = -x*(1j)*0.50

            syst = kwant.Builder() 
            model = make_system(type_graphene = 'monolayer')
            #model = make_syst_topo()
            area_per_site = np.abs(lat_const*lat_const*np.sqrt(3)/2)/2
            syst.fill(model, trunc, (0, 0));
            #syst[model.shape(trunc,(0,0))] = 0.
            #syst[model.neighbors()] = t1
            # add second neighbours hoppings
            #syst[model.a.neighbors()] = -t2
            #syst[model.b.neighbors()] = t2
            syst.eradicate_dangling()
            kwant.plot(syst);

            fsyst = syst.finalized()
        
            # Evaluate DOS
        
            rho = kwant.kpm.SpectralDensity(fsyst, num_moments = 100)
            energies, densities = rho() 
            
            # Evaluate conductivity tensor

            where = lambda s: np.linalg.norm(s.pos) < 1

            # xx component
        
            s_factory = kwant.kpm.LocalVectors(fsyst, where)
            cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x', num_vectors=None, vector_factory=s_factory)
            #cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x')
            
            cond_xx_miu = [cond_xx(mu = e, temperature = 0.01)/(area_per_site) for e in energies ]
            cond_xx_T = [cond_xx(mu = 0, temperature = T[i])/(area_per_site) for i in range(len(T)) ]

            # xy component

            s_factory = kwant.kpm.LocalVectors(fsyst, where)
            cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y', num_vectors=None, vector_factory=s_factory)
            #cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y')

            cond_xy_miu = [cond_xy(mu = e, temperature = 0.01)/(area_per_site) for e in energies ]
            cond_xy_T = [cond_xy(mu = 0, temperature = T[i])/(area_per_site) for i in range(len(T)) ] 
        
    
        ############
        # Plotting #
        ############

        save_fig_to = '/mnt/d/Masters/KPM/DOS/Haldane/'

        plt.plot(energies, densities)
        plt.title("Disordered Haldane DOS with KPM ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(np.abs(t2),2))+")", y=1.1)
        plt.xlim(-4,4)
        plt.ylim(0, 30000)
        plt.xlabel("Energy (E) / t")
        plt.ylabel(r"Density of states $\rho(E)$ (a.u.)")
        plt.tight_layout()
        #plt.margins(x = 1.5, y = 1.0)
        #plt.show()
    
        plt.savefig(save_fig_to + "DOS_dis_full_"+str(d)+"_tNNN_"+str(x)+".png", dpi='figure')
        plt.clf()

        # Conductivity tensor xx component

        plt.plot(energies, cond_xx_miu)
        plt.title("Zero temperature conductivity $\sigma_{xx}$ at different chemical potentials \n ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(np.abs(t2),2))+")", y=1.1)
        plt.xlabel(r"Chemical potential $\mu$")
        plt.ylabel(r"Conductivity $\sigma_{xx}$ / $e^2/h$")

        plt.tight_layout()
        plt.savefig(save_fig_to + "Sigma_xx_mu_dis_"+str(d)+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
        plt.clf()

        plt.plot(T, cond_xx_T)
        plt.title("Conductivity $\sigma_{xx}$ vs temperature for $\mu$ = 0 \n ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(np.abs(t2),2))+")", y=1.1)
        plt.xlabel(r"Temperature (T) eV$/k_B$")
        plt.ylabel(r"Conductivity $\sigma_{xx}$ / $e^2/h$")
        plt.xlim(-0.1,5.1)

        plt.tight_layout()
        plt.savefig(save_fig_to + "Sigma_xx_T_dis_"+str(d)+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
        plt.clf()

        # Conductivity tensor xy component

        plt.plot(energies, cond_xy_miu)
        plt.title("Zero temperature conductivity $\sigma_{xy}$ at different chemical potentials \n ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(np.abs(t2),2))+")", y=1.1)
        plt.xlabel(r"Chemical potential $\mu$")
        plt.ylabel(r"Conductivity $\sigma_{xy}$ / $e^2/h$")

        plt.tight_layout()
        plt.savefig(save_fig_to + "Sigma_xy_mu_dis_"+str(d)+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
        plt.clf()

        plt.plot(T, cond_xy_T)
        plt.title("Conductivity $\sigma_{xy}$ vs temperature for $\mu$ = 0 \n ($\sigma$ = "+str(round(d,2))+", t = "+str(round(t1, 2))+", t' = "+str(round(np.abs(t2),2))+")", y=1.1)
        plt.xlabel(r"Temperature (T) eV$/k_B$")
        plt.ylabel(r"Conductivity $\sigma_{xy}$ / $e^2/h$")
        plt.xlim(-0.1,5.1)

        plt.tight_layout()
        plt.savefig(save_fig_to + "Sigma_xy_T_dis_"+str(d)+"_tNNN_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
        plt.clf()
