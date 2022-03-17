import numpy as np
import matplotlib.pyplot as plt

disorders = [0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
miu = 0

for x in [0.00, 0.50]:

    for L in [200]:
    
        t1 = -1 + x
        t2 = x

        legend = []
        plot_xx = []
        plot_xy = []
        fig_cond_xx, ax_cond_xx = plt.subplots()
        fig_cond_xy, ax_cond_xy = plt.subplots()       

        e_ax_min = float("inf")
        e_ax_max = float("-inf")

        for d in disorders:

            est_energies = []
            est_densities = []
            est_cond_xx_miu = []
            est_cond_xy_miu = []

            file_name = "Haldane_DOS_dis_full_"+str(d)+"_tNNN_"+str(x)+".dat"

            with open(file_name) as file_fd:
                for line in file_fd.readlines():
                    est_energies.append(float(line.split()[0]))
                    est_cond_xx_miu.append(complex(line.split()[2]))
                    est_cond_xy_miu.append(complex(line.split()[3]))

            min_est_e = est_energies[0]
            max_est_e = est_energies[-1]
            if (min_est_e < e_ax_min):
                e_ax_min = min_est_e
            if (max_est_e > e_ax_max):
                e_ax_max = max_est_e


            save_fig_to = './'
                    
            # Conductance
        
            search = np.argwhere(np.array(est_energies) < miu)
            miu_point = len(search)+1
            plot_xx.append(np.complex(est_cond_xx_miu[miu_point]))
            plot_xy.append(np.complex(est_cond_xy_miu[miu_point]))

        #print(disorders)
        #print(plot)
        ax_cond_xx.plot(disorders, plot_xx)
        legend.append("$L$ = "+str("{:.0f}".format(L)))
        ax_cond_xx.set_title("Average conductivity $\sigma_{xx}$ at $\mu$ = "+str(round(miu, 2))+" as a function of disorder \n (|t| = "+str(round(np.abs(t1), 2))+", |t'| = "+str(round(np.abs(t2), 2))+")", y=1.1)
        ax_cond_xx.set_xlabel(r"Disorder strength $\sigma$")
        ax_cond_xx.set_ylabel(r"Conductivity $\langle \sigma_{xx} \rangle$ $(e^2/h)$")
        ax_cond_xx.set_xlim(int(disorders[0]), int(disorders[len(disorders)-1]))
        fig_cond_xx.legend(legend, loc='center left', bbox_to_anchor=(1.00, 0.50))       
        fig_cond_xx.tight_layout()
        fig_cond_xx.savefig(save_fig_to + "Gaussian_Scaling_Haldane_Sigma_xx_tNNN_"+str(x)+"_miu_"+str(miu)+".png", bbox_inches = 'tight', dpi='figure')


        ax_cond_xy.plot(disorders, plot_xy)
        legend.append("$L$ = "+str("{:.0f}".format(L)))
        ax_cond_xy.set_title("Average conductivity $\sigma_{xy}$ at $\mu$ = "+str(round(miu, 2))+" as a function of disorder \n (|t| = "+str(round(np.abs(t1), 2))+", |t'| = "+str(round(np.abs(t2), 2))+")", y=1.1)
        ax_cond_xy.set_xlabel(r"Disorder strength $\sigma$")
        ax_cond_xy.set_ylabel(r"Conductivity $\langle \sigma_{xy} \rangle$ $(e^2/h)$")
        ax_cond_xy.set_xlim(int(disorders[0]), int(disorders[len(disorders)-1]))
        fig_cond_xy.legend(legend, loc='center left', bbox_to_anchor=(1.00, 0.50))       
        fig_cond_xy.tight_layout()
        fig_cond_xy.savefig(save_fig_to + "Gaussian_Scaling_Haldane_Sigma_xy_tNNN_"+str(x)+"_miu_"+str(miu)+".png", bbox_inches = 'tight', dpi='figure')

