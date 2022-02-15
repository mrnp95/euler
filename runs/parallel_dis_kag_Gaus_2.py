import concurrent
import multiprocessing as mp
import time
import kwant
import numpy as np
from numpy import sqrt


##########
# BUILD  #
##########

def make_system(t1, t2, tn, lat_const, L, d, type_kagome='monolayer'):
    Bravais_vector = [(lat_const, 0),
                      (0.5 * lat_const, 0.5 * lat_const * sqrt(3))]  # Bravais vectors
    Bottom_Lat_pos = [(0.5 * lat_const, 0), (0.25 * lat_const, 0.25 * lat_const * sqrt(3)),
                      (0, 0)]  # The position of sublattice atoms in Bottom layer
    Upper_Lat_pos = [(0, 0), (-0.5 * lat_const, 0), (
    -0.25 * lat_const, -0.25 * lat_const * sqrt(3))]  # The position of sublattice atoms in Upper layer

    Bottom_lat = kwant.lattice.general(Bravais_vector, Bottom_Lat_pos, norbs=1)
    B_sub_a, B_sub_b, B_sub_c = Bottom_lat.sublattices

    if type_kagome == 'bilayer':
        Upper_lat = kwant.lattice.general(Bravais_vector, Upper_Lat_pos, norbs=1)
        U_sub_a, U_sub_b, U_sub_c = Upper_lat.sublattices

        # sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder()

    for x in range(int(-L / 2), int(L / 2)):
        for y in range(int(-L / 2), int(L / 2)):
            # define hopping and on-site potential on bottom layer kagome

            bulk[B_sub_a(x, y)] = np.random.normal(0, d, 1)
            bulk[B_sub_b(x, y)] = np.random.normal(0, d, 1)
            bulk[B_sub_c(x, y)] = np.random.normal(0, d, 1)

    bulk[kwant.builder.HoppingKind((0, 0), B_sub_b, B_sub_a)] = t1
    bulk[kwant.builder.HoppingKind((0, 0), B_sub_c, B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((0, 0), B_sub_a, B_sub_c)] = t1
    bulk[kwant.builder.HoppingKind((-1, 0), B_sub_a, B_sub_c)] = t1
    bulk[kwant.builder.HoppingKind((0, 1), B_sub_c, B_sub_b)] = t1
    bulk[kwant.builder.HoppingKind((1, -1), B_sub_b, B_sub_a)] = t1

    # Next neighbors

    bulk[kwant.builder.HoppingKind((0, 1), B_sub_c, B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((1, -1), B_sub_c, B_sub_a)] = t2
    # bulk[kwant.builder.HoppingKind((1,0), B_sub_b,B_sub_a)] = t2
    # bulk[kwant.builder.HoppingKind((0,-1), B_sub_b,B_sub_a)] = t2
    bulk[kwant.builder.HoppingKind((-1, 0), B_sub_a, B_sub_b)] = t2
    bulk[kwant.builder.HoppingKind((0, 1), B_sub_a, B_sub_b)] = t2
    # bulk[kwant.builder.HoppingKind((1,0), B_sub_c,B_sub_b)] = t2
    # bulk[kwant.builder.HoppingKind((-1,1), B_sub_c,B_sub_b)] = t2
    bulk[kwant.builder.HoppingKind((1, -1), B_sub_b, B_sub_c)] = t2
    bulk[kwant.builder.HoppingKind((-1, 0), B_sub_b, B_sub_c)] = t2
    # bulk[kwant.builder.HoppingKind((-1,1), B_sub_a,B_sub_c)] = t2
    # bulk[kwant.builder.HoppingKind((0,-1),B_sub_a,B_sub_c)] = t2

    # define hopping and on-site potential on upper layer kagome

    if type_kagome == 'bilayer':
        bulk[U_sub_a(0, 0)] = np.random.normal(0, 1, 1)
        bulk[U_sub_b(0, 0)] = np.random.normal(0, 1, 1)
        bulk[U_sub_c(0, 0)] = np.random.normal(0, 1, 1)

        bulk[kwant.builder.HoppingKind((0, 0), B_sub_a, B_sub_b)] = t1
        bulk[kwant.builder.HoppingKind((0, 0), B_sub_b, B_sub_c)] = t1
        bulk[kwant.builder.HoppingKind((0, 0), B_sub_c, B_sub_a)] = t1
        bulk[kwant.builder.HoppingKind((-1, 0), B_sub_a, B_sub_c)] = t1
        bulk[kwant.builder.HoppingKind((0, 1), B_sub_c, B_sub_b)] = t1
        bulk[kwant.builder.HoppingKind((1, -1), B_sub_b, B_sub_a)] = t1

        # Next neighbors

        bulk[kwant.builder.HoppingKind((0, 1), U_sub_c, U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((1, -1), U_sub_c, U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((1, 0), U_sub_b, U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((0, -1), U_sub_b, U_sub_a)] = t2
        bulk[kwant.builder.HoppingKind((-1, 0), U_sub_a, U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((0, 1), U_sub_a, U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((1, 0), U_sub_c, U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((-1, 1), U_sub_c, U_sub_b)] = t2
        bulk[kwant.builder.HoppingKind((1, -1), U_sub_b, U_sub_c)] = t2
        bulk[kwant.builder.HoppingKind((-1, 0), U_sub_b, U_sub_c)] = t2
        bulk[kwant.builder.HoppingKind((-1, 1), U_sub_a, U_sub_c)] = t2
        bulk[kwant.builder.HoppingKind((0, -1), U_sub_a, U_sub_c)] = t2

        bulk[kwant.builder.HoppingKind((0, 0), U_sub_a, B_sub_b)] = tn
        bulk[kwant.builder.HoppingKind((0, 0), U_sub_b, B_sub_c)] = tn
        bulk[kwant.builder.HoppingKind((0, 0), U_sub_c, B_sub_a)] = tn
        bulk[kwant.builder.HoppingKind((0, 0), U_sub_b, B_sub_a)] = tn
        bulk[kwant.builder.HoppingKind((0, 0), U_sub_c, B_sub_b)] = tn
        bulk[kwant.builder.HoppingKind((0, 0), U_sub_a, B_sub_c)] = tn

    return bulk


# Different geometries of the finite system

def trunc(site):
    x, y = abs(site.pos)
    return abs(x) < 800 and abs(y) < 800


def circle(site):
    x, y = site.pos
    r = 30
    return x ** 2 + y ** 2 < r ** 2


# Declare big things coming


def cluster_run(run_index):
    run_num = int(run_index[0])
    print(run_num)
    np.random.seed(run_num)
    print("DETAILS OF RUN WITH INDEX: " + str(run_num))
    print("################### Beginning of run " + str(run_num) + " ###################")

    # Default settings

    lat_const = 1  # lattice constant of kagome (unit: nm)
    tn = 0.0  # interlayer hopping between kagomes (unit: eV)
    L = 100  # size of the system (in each dimension)
    averaging = 200  # number of runs for averaging DOS and conductivities

    # Domains of cond function for later

    N_bins = 100  # Bins for energies in the estimator
    N_binsT = 500  # Bins for temperature
    T_min = 0.01
    T_max = 5.00
    T = np.linspace(T_min, T_max, N_binsT)

    x = run_index[1]
    d = run_index[2]

    print("|t'| = " + str(x) + ", d = " + str(d))

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
        model = make_system(t1, t2, tn, lat_const, L, d, type_kagome='monolayer')
        area_per_site = np.abs(lat_const * lat_const * np.sqrt(3) / 2) / 3
        syst.fill(model, trunc, (0, 0))

        syst.eradicate_dangling()

        # Plot system before running

        # kwant.plot(syst)

        fsyst = syst.finalized()

        # Evaluate DOS

        rho = kwant.kpm.SpectralDensity(fsyst)
        energies, densities = rho()
        print("Averaging:", av + 1, "/", averaging)

        # Evaluate conductivity tensor

        where = lambda s: np.linalg.norm(s.pos) < 1

        # xx component

        s_factory = kwant.kpm.LocalVectors(fsyst, where)
        cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x', num_vectors=None, vector_factory=s_factory)
        # cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x')

        cond_xx_miu = [cond_xx(mu=e, temperature=0.01) / (area_per_site) for e in energies]
        # print("-----mu------")
        # print([(cond_xx_miu[i], energies[i]) for i in range(0, len(energies))])
        # print()
        cond_xx_T = [cond_xx(mu=-1, temperature=T[i]) / (area_per_site) for i in range(len(T))]

        # xy component

        s_factory = kwant.kpm.LocalVectors(fsyst, where)
        cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y', num_vectors=None, vector_factory=s_factory)
        # cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y')

        cond_xy_miu = [cond_xy(mu=e, temperature=0.01) / (area_per_site) for e in energies]
        cond_xy_T = [cond_xy(mu=-1, temperature=T[i]) / (area_per_site) for i in range(len(T))]

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
        bin = int(np.floor((all_energies[i] - E_min) / (E_max - E_min) * N_bins))
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

    est_energies = [est_energies[i] / (bin_n[i]) for i in range(0, N_bins)]
    est_densities = [est_densities[i] / (bin_n[i]) for i in range(0, N_bins)]
    est_cond_xx_miu = [est_cond_xx_miu[i] / (bin_n[i]) for i in range(0, N_bins)]
    est_cond_xy_miu = [est_cond_xy_miu[i] / (bin_n[i]) for i in range(0, N_bins)]

    # Normalisation of DOS

    Norm_D = 0

    for i in range(0, len(est_energies)):
        Norm_D += est_densities[i] * (E_max - E_min) / N_bins

    est_densities = [est_densities[i] / Norm_D for i in range(0, N_bins)]

    # Conductivity as function of temperature

    for i in range(0, len(all_T)):
        bin = int(np.floor((all_T[i] - T_min) / (T_max - T_min) * N_binsT))
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

    est_cond_xx_T = [est_cond_xx_T[i] / (bin_nT[i]) for i in range(0, N_binsT)]
    est_cond_xy_T = [est_cond_xy_T[i] / (bin_nT[i]) for i in range(0, N_binsT)]

    ############
    # SAVING   #
    ############

    file_name = "Euler_DOS_dis_full_" + str(d) + "_tNNN_" + str(x) + ".dat"
    with open(file_name, "x") as file_fd:
        for i in range(0, len(est_energies)):
            file_fd.write(
                str(est_energies[i]) + " " + str(est_densities[i]) + " " + str(est_cond_xx_miu[i]) + " " + str(
                    est_cond_xy_miu[i]) + "\n")

    print("################### End of run " + str(run_num) + " ###################")
    return "Run " + str(run_num) + " Succeeded!"


def main():
    print("Number of processors: ", mp.cpu_count())

    xs = [0.0, 0.28, 0.33, 0.50, 1.0]
    ds = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

    row = len(xs) * len(ds)
    col = 3

    run_table = np.ones((row, col))
    cnt = 1
    for i in range(len(xs)):
        for j in range(len(ds)):
            run_table[i * len(ds) + j, 0] = cnt
            run_table[i * len(ds) + j, 1] = xs[i]
            run_table[i * len(ds) + j, 2] = ds[j]
            cnt += 1

    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(cluster_run, run_table)

        for result in results:
            print(result)

    finish = time.perf_counter()

    print("Results:\n", results, f" Time period: {round(finish - start, 2)}")


if __name__ == '__main__':
    main()

        


