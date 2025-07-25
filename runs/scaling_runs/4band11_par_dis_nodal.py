import concurrent
import multiprocessing as mp
import time
import kwant
import numpy as np
from numpy import sqrt


##########
# BUILD  #
##########

def make_system(lat_const, L, d, delta=0.50):
    Bravais_vector = [(lat_const, 0),
                      (0, lat_const)]  # Bravais vectors
    Bottom_Lat_pos = [(0, 0), (0, 0), (0, 0), (0, 0)]  # The position of sublattice atoms in Bottom layer
    Bottom_lat = kwant.lattice.general(Bravais_vector, Bottom_Lat_pos, norbs=1)
    B_sub_A, B_sub_B, B_sub_C, B_sub_D = Bottom_lat.sublattices

    sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1])
    bulk = kwant.Builder(sym)

    # define hoppings and on-site potentials

    for x in range(int(-L / 2), int(L / 2)):
        for y in range(int(-L / 2), int(L / 2)):
            bulk[B_sub_A(x, y)] = 0.4036 + delta + np.random.normal(0, d, 1)
            bulk[B_sub_B(x, y)] = 0.4036 - delta + np.random.normal(0, d, 1)
            bulk[B_sub_C(x, y)] = -0.4036 - delta + np.random.normal(0, d, 1)
            bulk[B_sub_D(x, y)] = -0.4036 + delta + np.random.normal(0, d, 1)

    bulk[kwant.builder.HoppingKind((-2, -2), B_sub_A, B_sub_A)] = -0.0278
    # bulk[kwant.builder.HoppingKind((-2,-1), B_sub_A,B_sub_A)] = 0
    bulk[kwant.builder.HoppingKind((-2, 0), B_sub_A, B_sub_A)] = 0.0456
    # bulk[kwant.builder.HoppingKind((-2,1), B_sub_A,B_sub_A)] = 0
    bulk[kwant.builder.HoppingKind((-2, 2), B_sub_A, B_sub_A)] = -0.0278
    # bulk[kwant.builder.HoppingKind((-1,-2), B_sub_A,B_sub_A)] = 0
    bulk[kwant.builder.HoppingKind((-1, -1), B_sub_A, B_sub_A)] = -0.1356
    bulk[kwant.builder.HoppingKind((-1, 0), B_sub_A, B_sub_A)] = -0.25
    bulk[kwant.builder.HoppingKind((-1, 1), B_sub_A, B_sub_A)] = -0.1356
    # bulk[kwant.builder.HoppingKind((-1,2), B_sub_A,B_sub_A)] = 0
    bulk[kwant.builder.HoppingKind((0, -2), B_sub_A, B_sub_A)] = 0.0456
    bulk[kwant.builder.HoppingKind((0, -1), B_sub_A, B_sub_A)] = -0.25
    # bulk[kwant.builder.HoppingKind((0,0), B_sub_A,B_sub_A)] = 0.4036
    bulk[kwant.builder.HoppingKind((0, 1), B_sub_A, B_sub_A)] = -0.25
    bulk[kwant.builder.HoppingKind((0, 2), B_sub_A, B_sub_A)] = 0.0456

    bulk[kwant.builder.HoppingKind((-2, -2), B_sub_B, B_sub_B)] = -0.0278
    # bulk[kwant.builder.HoppingKind((-2,-1), B_sub_B,B_sub_B)] = 0
    bulk[kwant.builder.HoppingKind((-2, 0), B_sub_B, B_sub_B)] = 0.0456
    # bulk[kwant.builder.HoppingKind((-2,1), B_sub_B,B_sub_B)] = 0
    bulk[kwant.builder.HoppingKind((-2, 2), B_sub_B, B_sub_B)] = -0.0278
    # bulk[kwant.builder.HoppingKind((-1,-2), B_sub_B,B_sub_B)] = 0
    bulk[kwant.builder.HoppingKind((-1, -1), B_sub_B, B_sub_B)] = -0.1356
    bulk[kwant.builder.HoppingKind((-1, 0), B_sub_B, B_sub_B)] = -0.25
    bulk[kwant.builder.HoppingKind((-1, 1), B_sub_B, B_sub_B)] = -0.1356
    # bulk[kwant.builder.HoppingKind((-1,2), B_sub_B,B_sub_B)] = 0
    bulk[kwant.builder.HoppingKind((0, -2), B_sub_B, B_sub_B)] = 0.0456
    bulk[kwant.builder.HoppingKind((0, -1), B_sub_B, B_sub_B)] = -0.25
    # bulk[kwant.builder.HoppingKind((0,0), B_sub_B,B_sub_B)] = 0.4036
    bulk[kwant.builder.HoppingKind((0, 1), B_sub_B, B_sub_B)] = -0.25
    bulk[kwant.builder.HoppingKind((0, 2), B_sub_B, B_sub_B)] = 0.0456

    bulk[kwant.builder.HoppingKind((-2, -2), B_sub_C, B_sub_C)] = 0.0278
    # bulk[kwant.builder.HoppingKind((-2,-1), B_sub_C,B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((-2, 0), B_sub_C, B_sub_C)] = -0.0456
    # bulk[kwant.builder.HoppingKind((-2,1), B_sub_C,B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((-2, 2), B_sub_C, B_sub_C)] = 0.0278
    # bulk[kwant.builder.HoppingKind((-1,-2), B_sub_C,B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((-1, -1), B_sub_C, B_sub_C)] = 0.1356
    bulk[kwant.builder.HoppingKind((-1, 0), B_sub_C, B_sub_C)] = 0.25
    bulk[kwant.builder.HoppingKind((-1, 1), B_sub_C, B_sub_C)] = 0.1356
    # bulk[kwant.builder.HoppingKind((-1,2), B_sub_C,B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((0, -2), B_sub_C, B_sub_C)] = -0.0456
    bulk[kwant.builder.HoppingKind((0, -1), B_sub_C, B_sub_C)] = 0.25
    # bulk[kwant.builder.HoppingKind((0,0), B_sub_C,B_sub_C)] =  -0.4036
    bulk[kwant.builder.HoppingKind((0, 1), B_sub_C, B_sub_C)] = 0.25
    bulk[kwant.builder.HoppingKind((0, 2), B_sub_C, B_sub_C)] = -0.0456

    bulk[kwant.builder.HoppingKind((-2, -2), B_sub_D, B_sub_D)] = 0.0278
    # bulk[kwant.builder.HoppingKind((-2,-1), B_sub_D,B_sub_D)] = 0
    bulk[kwant.builder.HoppingKind((-2, 0), B_sub_D, B_sub_D)] = -0.0456
    # bulk[kwant.builder.HoppingKind((-2,1), B_sub_D,B_sub_D)] = 0
    bulk[kwant.builder.HoppingKind((-2, 2), B_sub_D, B_sub_D)] = 0.0278
    # bulk[kwant.builder.HoppingKind((-1,-2), B_sub_D,B_sub_D)] = 0
    bulk[kwant.builder.HoppingKind((-1, -1), B_sub_D, B_sub_D)] = 0.1356
    bulk[kwant.builder.HoppingKind((-1, 0), B_sub_D, B_sub_D)] = 0.25
    bulk[kwant.builder.HoppingKind((-1, 1), B_sub_D, B_sub_D)] = 0.1356
    # bulk[kwant.builder.HoppingKind((-1,2), B_sub_D,B_sub_D)] = 0
    bulk[kwant.builder.HoppingKind((0, -2), B_sub_D, B_sub_D)] = -0.0456
    bulk[kwant.builder.HoppingKind((0, -1), B_sub_D, B_sub_D)] = 0.25
    # bulk[kwant.builder.HoppingKind((0,0), B_sub_D,B_sub_D)] =  -0.4036
    bulk[kwant.builder.HoppingKind((0, 1), B_sub_D, B_sub_D)] = 0.25
    bulk[kwant.builder.HoppingKind((0, 2), B_sub_D, B_sub_D)] = -0.0456

    bulk[kwant.builder.HoppingKind((-2, -2), B_sub_A, B_sub_C)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-2, -1), B_sub_A, B_sub_C)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-2, 0), B_sub_A, B_sub_C)] = 0.0348 * 1j
    bulk[kwant.builder.HoppingKind((-2, 1), B_sub_A, B_sub_C)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-2, 2), B_sub_A, B_sub_C)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-1, -2), B_sub_A, B_sub_C)] = 0.0234 * 1j
    bulk[kwant.builder.HoppingKind((-1, -1), B_sub_A, B_sub_C)] = -0.09635 * 1j
    bulk[kwant.builder.HoppingKind((-1, 0), B_sub_A, B_sub_C)] = -0.3204 * 1j
    bulk[kwant.builder.HoppingKind((-1, 1), B_sub_A, B_sub_C)] = -0.09635 * 1j
    bulk[kwant.builder.HoppingKind((-1, 2), B_sub_A, B_sub_C)] = 0.0234 * 1j
    # bulk[kwant.builder.HoppingKind((0,-2), B_sub_A,B_sub_C)] = 0
    # bulk[kwant.builder.HoppingKind((0,-1), B_sub_A,B_sub_C)] = 0
    # bulk[kwant.builder.HoppingKind((0,0), B_sub_A,B_sub_C)] = 0
    # bulk[kwant.builder.HoppingKind((0,1), B_sub_A,B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((0, 2), B_sub_A, B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((1, -2), B_sub_A, B_sub_C)] = -0.0234 * 1j
    bulk[kwant.builder.HoppingKind((1, -1), B_sub_A, B_sub_C)] = 0.09635 * 1j
    bulk[kwant.builder.HoppingKind((1, 0), B_sub_A, B_sub_C)] = 0.3204 * 1j
    bulk[kwant.builder.HoppingKind((1, 1), B_sub_A, B_sub_C)] = 0.09635 * 1j
    bulk[kwant.builder.HoppingKind((1, 2), B_sub_A, B_sub_C)] = -0.0234 * 1j
    bulk[kwant.builder.HoppingKind((2, -2), B_sub_A, B_sub_C)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((2, -1), B_sub_A, B_sub_C)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((2, 0), B_sub_A, B_sub_C)] = -0.0348 * 1j
    bulk[kwant.builder.HoppingKind((2, 1), B_sub_A, B_sub_C)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((2, 2), B_sub_A, B_sub_C)] = 0.0063 * 1j

    bulk[kwant.builder.HoppingKind((-2, -2), B_sub_A, B_sub_D)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-2, -1), B_sub_A, B_sub_D)] = -0.0234 * 1j
    # bulk[kwant.builder.HoppingKind((-2,0), B_sub_A,B_sub_D)] = 0
    bulk[kwant.builder.HoppingKind((-2, 1), B_sub_A, B_sub_D)] = 0.0234 * 1j
    bulk[kwant.builder.HoppingKind((-2, 2), B_sub_A, B_sub_D)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-1, -2), B_sub_A, B_sub_D)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-1, -1), B_sub_A, B_sub_D)] = 0.09635 * 1j
    # bulk[kwant.builder.HoppingKind((-1,0), B_sub_A,B_sub_D)] = 0
    bulk[kwant.builder.HoppingKind((-1, 1), B_sub_A, B_sub_D)] = -0.09635 * 1j
    bulk[kwant.builder.HoppingKind((-1, 2), B_sub_A, B_sub_D)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((0, -2), B_sub_A, B_sub_D)] = -0.0348 * 1j
    bulk[kwant.builder.HoppingKind((0, -1), B_sub_A, B_sub_D)] = 0.3204 * 1j
    # bulk[kwant.builder.HoppingKind((0,0), B_sub_A,B_sub_D)] = 0
    bulk[kwant.builder.HoppingKind((0, 1), B_sub_A, B_sub_D)] = -0.3204 * 1j
    bulk[kwant.builder.HoppingKind((0, 2), B_sub_A, B_sub_D)] = 0.0348 * 1j
    bulk[kwant.builder.HoppingKind((1, -2), B_sub_A, B_sub_D)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((1, -1), B_sub_A, B_sub_D)] = 0.09635 * 1j
    # bulk[kwant.builder.HoppingKind((1,0), B_sub_A,B_sub_D)] = 0
    bulk[kwant.builder.HoppingKind((1, 1), B_sub_A, B_sub_D)] = -0.09635 * 1j
    bulk[kwant.builder.HoppingKind((1, 2), B_sub_A, B_sub_D)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((2, -2), B_sub_A, B_sub_D)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((2, -1), B_sub_A, B_sub_D)] = -0.0234 * 1j
    # bulk[kwant.builder.HoppingKind((2,0), B_sub_A,B_sub_D)] = 0
    bulk[kwant.builder.HoppingKind((2, 1), B_sub_A, B_sub_D)] = 0.0234 * 1j
    bulk[kwant.builder.HoppingKind((2, 2), B_sub_A, B_sub_D)] = -0.0063 * 1j

    bulk[kwant.builder.HoppingKind((-2, -2), B_sub_B, B_sub_C)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-2, -1), B_sub_B, B_sub_C)] = -0.0234 * 1j
    # bulk[kwant.builder.HoppingKind((-2,0), B_sub_B,B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((-2, 1), B_sub_B, B_sub_C)] = 0.0234 * 1j
    bulk[kwant.builder.HoppingKind((-2, 2), B_sub_B, B_sub_C)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-1, -2), B_sub_B, B_sub_C)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-1, -1), B_sub_B, B_sub_C)] = 0.09635 * 1j
    # bulk[kwant.builder.HoppingKind((-1,0), B_sub_B,B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((-1, 1), B_sub_B, B_sub_C)] = -0.09635 * 1j
    bulk[kwant.builder.HoppingKind((-1, 2), B_sub_B, B_sub_C)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((0, -2), B_sub_B, B_sub_C)] = -0.0348 * 1j
    bulk[kwant.builder.HoppingKind((0, -1), B_sub_B, B_sub_C)] = 0.3204 * 1j
    # bulk[kwant.builder.HoppingKind((0,0), B_sub_B,B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((0, 1), B_sub_B, B_sub_C)] = -0.3204 * 1j
    bulk[kwant.builder.HoppingKind((0, 2), B_sub_B, B_sub_C)] = 0.0348 * 1j
    bulk[kwant.builder.HoppingKind((1, -2), B_sub_B, B_sub_C)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((1, -1), B_sub_B, B_sub_C)] = 0.09635 * 1j
    # bulk[kwant.builder.HoppingKind((1,0), B_sub_B,B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((1, 1), B_sub_B, B_sub_C)] = -0.09635 * 1j
    bulk[kwant.builder.HoppingKind((1, 2), B_sub_B, B_sub_C)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((2, -2), B_sub_B, B_sub_C)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((2, -1), B_sub_B, B_sub_C)] = -0.0234 * 1j
    # bulk[kwant.builder.HoppingKind((2,0), B_sub_B,B_sub_C)] = 0
    bulk[kwant.builder.HoppingKind((2, 1), B_sub_B, B_sub_C)] = 0.0234 * 1j
    bulk[kwant.builder.HoppingKind((2, 2), B_sub_B, B_sub_C)] = -0.0063 * 1j

    bulk[kwant.builder.HoppingKind((-2, -2), B_sub_B, B_sub_D)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-2, -1), B_sub_B, B_sub_D)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-2, 0), B_sub_B, B_sub_D)] = -0.0348 * 1j
    bulk[kwant.builder.HoppingKind((-2, 1), B_sub_B, B_sub_D)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-2, 2), B_sub_B, B_sub_D)] = 0.0063 * 1j
    bulk[kwant.builder.HoppingKind((-1, -2), B_sub_B, B_sub_D)] = -0.0234 * 1j
    bulk[kwant.builder.HoppingKind((-1, -1), B_sub_B, B_sub_D)] = 0.09635 * 1j
    bulk[kwant.builder.HoppingKind((-1, 0), B_sub_B, B_sub_D)] = 0.3204 * 1j
    bulk[kwant.builder.HoppingKind((-1, 1), B_sub_B, B_sub_D)] = 0.09635 * 1j
    bulk[kwant.builder.HoppingKind((-1, 2), B_sub_B, B_sub_D)] = -0.0234 * 1j
    # bulk[kwant.builder.HoppingKind((0,-2), B_sub_B,B_sub_D)] = 0
    # bulk[kwant.builder.HoppingKind((0,-1), B_sub_B,B_sub_D)] = 0
    # bulk[kwant.builder.HoppingKind((0,0), B_sub_B,B_sub_D)] = 0
    # bulk[kwant.builder.HoppingKind((0,1), B_sub_B,B_sub_D)] = 0
    # bulk[kwant.builder.HoppingKind((0,2), B_sub_B,B_sub_D)] = 0
    bulk[kwant.builder.HoppingKind((1, -2), B_sub_B, B_sub_D)] = 0.0234 * 1j
    bulk[kwant.builder.HoppingKind((1, -1), B_sub_B, B_sub_D)] = -0.09635 * 1j
    bulk[kwant.builder.HoppingKind((1, 0), B_sub_B, B_sub_D)] = -0.3204 * 1j
    bulk[kwant.builder.HoppingKind((1, 1), B_sub_B, B_sub_D)] = -0.09635 * 1j
    bulk[kwant.builder.HoppingKind((1, 2), B_sub_B, B_sub_D)] = 0.0234 * 1j
    bulk[kwant.builder.HoppingKind((2, -2), B_sub_B, B_sub_D)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((2, -1), B_sub_B, B_sub_D)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((2, 0), B_sub_B, B_sub_D)] = 0.0348 * 1j
    bulk[kwant.builder.HoppingKind((2, 1), B_sub_B, B_sub_D)] = -0.0063 * 1j
    bulk[kwant.builder.HoppingKind((2, 2), B_sub_B, B_sub_D)] = -0.0063 * 1j

    return bulk


# Different geometries of the finite system

def trunc(site):
    x, y = abs(site.pos)
    return abs(x) < 51 and abs(y) < 51


def circle(site):
    x, y = site.pos
    r = 30
    return x ** 2 + y ** 2 < r ** 2


def cluster_run(run_index):
    run_num = int(run_index[0])
    print(run_num)
    np.random.seed(run_num)
    print("DETAILS OF RUN WITH INDEX: " + str(run_num))
    print("################### Beginning of run " + str(run_num) + " ###################")

    # Default settings

    lat_const = 1  # lattice constant of kagome (unit: nm)
    L = 60  # size of the system (in each dimension) 60 - 100
    averaging = 5  # number of runs for averaging DOS and conductivities --> 300

    # Domains of cond function for later

    N_bins = 200  # Bins for energies in the estimator
    N_binsT = 500  # Bins for temperature
    T_min = 0.01
    T_max = 5.00
    T = np.linspace(T_min, T_max, N_binsT)

    x = run_index[1]
    d = run_index[2]

    print("m = " + str(x) + ", d = " + str(d))

    m = x

    all_energies = []
    all_densities = []
    all_cond_xx_miu = []
    all_cond_xy_miu = []
    all_T = []
    all_cond_xx_T = []
    all_cond_xy_T = []

    for av in range(0, averaging):
        # Hoppings

        syst = kwant.Builder()
        model = make_system(lat_const, L, d, delta=0.50)
        area_per_site = np.abs(lat_const * lat_const)
        syst.fill(model, trunc, (0, 0));

        syst.eradicate_dangling()

        # Plot system before running

        # kwant.plot(syst);

        fsyst = syst.finalized()

        # Evaluate DOS

        rho = kwant.kpm.SpectralDensity(fsyst, num_moments=4096, num_vectors=1000)
        energies, densities = rho()
        print("Averaging:", av + 1, "/", averaging)

        # Evaluate conductivity tensor

        where = lambda s: np.linalg.norm(s.pos) < 1

        # xx component

        s_factory = kwant.kpm.LocalVectors(fsyst, where)
        cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x', num_vectors=None, vector_factory=s_factory)
        # cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x')

        cond_xx_miu = [cond_xx(mu=e, temperature=0.00) / (area_per_site) for e in energies]
        # print("-----mu------")
        # print([(cond_xx_miu[i], energies[i]) for i in range(0, len(energies))])
        # print()
        cond_xx_T = [cond_xx(mu=-1, temperature=T[i]) / (area_per_site) for i in range(len(T))]

        # xy component

        s_factory = kwant.kpm.LocalVectors(fsyst, where)
        cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y', num_vectors=None, vector_factory=s_factory)
        # cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y')

        cond_xy_miu = [cond_xy(mu=e, temperature=0.00) / (area_per_site) for e in energies]
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

    file_name = f"../data/scaling/L_{L}/4band11_DOS_dis_full_" + str(d) + "_m_" + str(x) + "_L_" + str(L) + ".dat"
    with open(file_name, "x") as file_fd:
        for i in range(0, len(est_energies)):
            file_fd.write(
                str(est_energies[i]) + " " + str(est_densities[i]) + " " + str(est_cond_xx_miu[i]) + " " + str(
                    est_cond_xy_miu[i]) + "\n")

    print("################### End of run " + str(run_num) + " ###################")
    return "Run " + str(run_num) + " Succeeded!"


def main():
    print("Number of processors: ", mp.cpu_count())

    xs = [1.0]
    # ds = [0.00, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    ds = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

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




