import numpy as np
import matplotlib.pyplot as plt

disorders = [0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
miu = 0

for x in [0.00, 0.28, 0.33, 0.50, 1.00]:

    legend = []
    fig_cond_xx, ax_cond_xx = plt.subplots()

    for L in [50]:
        
        plot = []

        t1 = 0.25
        t2 = -0.25
        t3 = x
     
        e_ax_min = float("inf")
        e_ax_max = float("-inf")

        for d in disorders:
        
            est_energies = []
            est_densities = []
            est_cond_xx_miu = []
            est_cond_xy_miu = []

            file_name = "Euler_DOS_dis_full_"+str(d)+"_tNNNN_"+str(x)+"_L_"+str(L)+".dat"

            with open(file_name) as file_fd:
                for line in file_fd.readlines():
                    est_energies.append(float(line.split()[0]))
                    est_cond_xx_miu.append(complex(line.split()[2]))
        
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
            plot.append(np.complex(est_cond_xx_miu[miu_point]))
        
        #print(disorders)
        #print(plot)
        ax_cond_xx.plot(disorders, plot)
        legend.append("$L$ = "+str("{:.0f}".format(L)))
        ax_cond_xx.set_title("Average conductivity $\sigma_{xx}$ at $\mu$ = "+str(round(miu, 2))+" as a function of disorder \n (t = "+str(round(t1, 2))+", t' = "+str(round(t2, 2))+", t'' = "+str(round(t3, 2))+")", y=1.1)
        ax_cond_xx.set_xlabel(r"Disorder strength $\sigma$")
        ax_cond_xx.set_ylabel(r"Conductivity $\langle \sigma_{xx} \rangle$ $(e^2/h)$")
        ax_cond_xx.set_xlim(int(disorders[0]), int(disorders[len(disorders)-1]))
    fig_cond_xx.legend(legend, loc='upper right', bbox_to_anchor=(0.90, 0.8))       
    fig_cond_xx.tight_layout()
    fig_cond_xx.savefig(save_fig_to + "Gaussian_Scaling_Euler_Sigma_xx_tNNNN_"+str(x)+"_miu_"+str(miu)+".png", bbox_inches = 'tight', dpi='figure')
       
