import numpy as np
import matplotlib.pyplot as plt


for x in [1.00]:
    
    legend = []
    fig_dos, ax_dos = plt.subplots()
    fig_cond_xx, ax_cond_xx = plt.subplots()
    fig_cond_xy, ax_cond_xy = plt.subplots()
    e_ax_min = float("inf")
    e_ax_max = float("-inf")

    for d in [0.00]: #[0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:

        est_energies = []
        est_densities = []
        est_cond_xx_miu = []
        est_cond_xy_miu = []

        file_name = "4band11_DOS_dis_full_"+str(d)+"_m_"+str(x)+"_L_50.dat"

        with open(file_name) as file_fd:
            for line in file_fd.readlines():
                est_energies.append(float(line.split()[0]))
                est_densities.append(complex(line.split()[1]))
                est_cond_xx_miu.append(complex(line.split()[2]))
                est_cond_xy_miu.append(complex(line.split()[3]))
        
        
        min_est_e = est_energies[0]
        max_est_e = est_energies[-1]
        if (min_est_e < e_ax_min):
            e_ax_min = min_est_e
        if (max_est_e > e_ax_max):
            e_ax_max = max_est_e


        save_fig_to = './'
        legend.append("$\sigma$ = "+str("{:.2f}".format(d)))

        ax_dos.plot(est_energies, est_densities)
        ax_dos.set_title("Disordered four band Euler DOS with KPM (m = "+str(round(x, 2))+")", y=1.1)
        ax_dos.set_xlabel("Energy (E) / hopping unit")
        ax_dos.set_ylabel(r"Density of states $\rho(E)$ (a.u.)")
                
        # Conductivity tensor xx component
        
        ax_cond_xx.plot(est_energies, est_cond_xx_miu)
        ax_cond_xx.set_title("Average conductivity $\sigma_{xx}$ at different chemical potentials \n (m = "+str(round(x, 2))+")", y=1.1)
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
        ax_cond_xy.set_title("Average conductivity $\sigma_{xy}$ at different chemical potentials \n (m = "+str(round(x, 2))+")", y=1.1)
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
    ax_dos.set_ylim(0, 3.0)
    fig_dos.legend(legend, loc='center left', bbox_to_anchor=(1.00, 0.5))
    fig_cond_xx.legend(legend, loc='center left', bbox_to_anchor=(1.00, 0.5))
    fig_cond_xy.legend(legend, loc='center left', bbox_to_anchor=(1.00, 0.5))
    fig_dos.tight_layout() 
    fig_cond_xx.tight_layout() 
    fig_cond_xy.tight_layout() 
    fig_dos.savefig(save_fig_to + "4band_DOS_dis_full_m_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
    fig_cond_xx.savefig(save_fig_to + "4band_Sigma_xx_mu_m_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
    fig_cond_xy.savefig(save_fig_to + "4band_Sigma_xy_mu_m_"+str(x)+".png", bbox_inches = 'tight', dpi='figure')
