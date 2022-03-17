import kwant 
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.linalg as la
from numpy import sqrt
import tinyarray as ta
import matplotlib.pyplot as plt


lat_const = 1  # lattice constant of square lattice (unit: nm)
tA = 0.35    # hoppings (unit: eV)
tB = 0.46
tC = 0.69    
epsA = 0
epsB = 0
epsC = 0

d  = 1.0    # standard deviation in Gaussian disorder (unit: eV)
L = 30 # size of the system (in each dimension)
averaging = 3 # number of runs for averaging DOS and conductivities

# Domains of cond function for later

N_bins = 100 # Bins for energies in the estimator
N_binsT = 500 # Bins for temperature
T_min = 0.01
T_max = 5.00
T = np.linspace(T_min, T_max, N_binsT)


# Free parameters 

delta0 = 1
delta1 = 1
delta2 = 1
delta3 = 1
delta4 = 1
delta5 = 1
delta6 = 1
delta7 = 1
delta8 = 1

beta0 = 1
beta1 = 1
beta2 = 1
beta3 = 1
beta4 = 1
beta5 = 1
beta6 = 1
beta7 = 1
beta8 = 1

# Hopping parameters (for embedding)

a_00 = epsA
a_01 = 0
a_02 = 0
a_03 = 2*tA
a_04 = 0
a_05 = 0
a_06 = 0
a_07 = 2*tA
a_08 = 0

b_00 = epsB
b_01 = 2*tA
b_02 = 0
b_03 = 0
b_04 = 0
b_05 = 2*tA
b_06 = 0
b_07 = 0
b_08 = 0

c_00 = epsC
c_01 = -2*tA
c_02 = np.conj(b_06)
c_03 = -2*tA
c_04 = np.conj(b_08)
c_05 = -2*tA
c_06 = np.conj(b_02)
c_07 = np.conj(b_03)
c_08 = -2*tA

d_00 = 0
d_01 = 0
d_02 = -1/2*tB
d_03 = 0
d_04 = 1/2*tB
d_05 = 0
d_06 = -1/2*tB
d_07 = 0
d_08 = 1/2*tB

e_00 = np.conj(d_00)
e_01 = np.conj(d_05)
e_02 = np.conj(d_06)
e_03 = np.conj(d_07)
e_04 = np.conj(d_08)
e_05 = np.conj(d_01)
e_06 = np.conj(d_02)
e_07 = np.conj(d_03)
e_08 = np.conj(d_04)

f_00 = 0
f_01 = 0
f_02 = 0
f_03 = 0
f_04 = 0
f_05 = 0
f_06 = 0
f_07 = 0
f_08 = 0

# Upper layer

a_u0 = a_00/2
a_u1 = a_01/2
a_u2 = a_02/2
a_u3 = a_03/2
a_u4 = a_04/2
a_u5 = a_05/2
a_u6 = a_06/2
a_u7 = a_07/2
a_u8 = a_08/2

b_u0 = b_00/2
b_u1 = (1-beta1)*b_01
b_u2 = (1-beta2)*b_02
b_u3 = (1-beta3)*b_03
b_u4 = (1-beta4)*b_04
b_u5 = (1-beta5)*b_05
b_u6 = (1-beta6)*b_06
b_u7 = (1-beta7)*b_07
b_u8 = (1-beta8)*b_08

c_u0 = np.conj(b_u0)
c_u1 = np.conj(b_u5)
c_u2 = np.conj(b_u6)
c_u3 = np.conj(b_u7)
c_u4 = np.conj(b_u8)
c_u5 = np.conj(b_u1)
c_u6 = np.conj(b_u2)
c_u7 = np.conj(b_u3)
c_u8 = np.conj(b_u4)

d_u0 = -(1-delta0)*d_00
d_u1 = (1-delta1)*d_01
d_u2 = (1-delta2)*d_02
d_u3 = (1-delta3)*d_03
d_u4 = (1-delta4)*d_04
d_u5 = (1-delta5)*d_05
d_u6 = (1-delta6)*d_06
d_u7 = (1-delta7)*d_07
d_u8 = (1-delta8)*d_08

e_u0 = np.conj(d_u0) #check this one
e_u1 = np.conj(d_u5)
e_u2 = np.conj(d_u6)
e_u3 = np.conj(d_u7)
e_u4 = np.conj(d_u8)
e_u5 = np.conj(d_u1)
e_u6 = np.conj(d_u2)
e_u7 = np.conj(d_u3)
e_u8 = np.conj(d_u4)

f_u0 = f_00/2
f_u1 = f_01
f_u2 = f_02
f_u3 = f_03
f_u4 = f_04
f_u5 = f_05
f_u6 = f_06
f_u7 = f_07
f_u8 = f_08

# Bottom layer

a_d0 = a_u0
a_d1 = a_u1
a_d2 = a_u2
a_d3 = a_u3
a_d4 = a_u4
a_d5 = a_u5
a_d6 = a_u6
a_d7 = a_u7
a_d8 = a_u8

b_d0 = np.conj(b_u0)
b_d1 = np.conj(b_u1)
b_d2 = np.conj(b_u2)
b_d3 = np.conj(b_u3)
b_d4 = np.conj(b_u8)
b_d5 = np.conj(b_u1)
b_d6 = np.conj(b_u2)
b_d7 = np.conj(b_u3)
b_d8 = np.conj(b_u4)

c_d0 = np.conj(b_d0)
c_d1 = np.conj(b_d5)
c_d2 = np.conj(b_d6)
c_d3 = np.conj(b_d7)
c_d4 = np.conj(b_d8)
c_d5 = np.conj(b_d1)
c_d6 = np.conj(b_d2)
c_d7 = np.conj(b_d3)
c_d8 = np.conj(b_d4)

d_d0 = -delta0*d_00
d_d1 = -delta1*d_01/2
d_d2 = -delta2*d_02/2
d_d3 = -delta3*d_03/2
d_d4 = -delta4*d_04/2
d_d5 = -delta5*d_05/2
d_d6 = -delta6*d_06/2
d_d7 = -delta7*d_07/2
d_d8 = -delta8*d_08/2

e_d0 = np.conj(d_d0) #check this one
e_d1 = np.conj(d_d5)
e_d2 = np.conj(d_d6)
e_d3 = np.conj(d_d7)
e_d4 = np.conj(d_d8)
e_d5 = np.conj(d_d1)
e_d6 = np.conj(d_d2)
e_d7 = np.conj(d_d3)
e_d8 = np.conj(d_d4)

f_d0 = f_u0
f_d1 = f_u1
f_d2 = f_u2
f_d3 = f_u3
f_d4 = f_u4
f_d5 = f_u5
f_d6 = f_u6
f_d7 = f_u7
f_d8 = f_u8


def make_system(type_embedded_C4 = 'multilayer'):
    
    Bravais_vector = [(lat_const, 0, 0), 
                        (0,  lat_const, 0),
                        (0, 0, lat_const)] # Bravais vectors
    Lat_pos = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]    # The position of sublattice atoms in Bottom layer  
    Lat = kwant.lattice.general(Bravais_vector, Lat_pos, norbs=1)
    B_sub_a, B_sub_b, B_sub_c = Lat.sublattices 
    
    sym = kwant.TranslationalSymmetry(Bravais_vector[0], Bravais_vector[1], Bravais_vector[2])
    bulk = kwant.Builder()

    # define hoppings and on-site potentials
    
    for x in range(int(-L/2), int(L/2)):
            for y in range(int(-L/2), int(L/2)):

                bulk[B_sub_a(x,y,0)] = epsA+np.random.normal(0, d, 1)
                bulk[B_sub_b(x,y,0)] = epsB+np.random.normal(0, d, 1)
                bulk[B_sub_c(x,y,0)] = epsC+np.random.normal(0, d, 1)
                
                bulk[B_sub_a(x,y,1)] = epsA+np.random.normal(0, d, 1)
                bulk[B_sub_b(x,y,1)] = epsB+np.random.normal(0, d, 1)
                bulk[B_sub_c(x,y,1)] = epsC+np.random.normal(0, d, 1)
                
                bulk[B_sub_a(x,y,-1)] = epsA+np.random.normal(0, d, 1)
                bulk[B_sub_b(x,y,-1)] = epsB+np.random.normal(0, d, 1)
                bulk[B_sub_c(x,y,-1)] = epsC+np.random.normal(0, d, 1)

    bulk[kwant.builder.HoppingKind((0,1,0), B_sub_a,B_sub_a)] =  2*tA #a_03
    bulk[kwant.builder.HoppingKind((0,-1,0), B_sub_a,B_sub_a)] = 2*tA #a_07
    bulk[kwant.builder.HoppingKind((1,0,0), B_sub_b,B_sub_b)] = 2*tA #b_01
    bulk[kwant.builder.HoppingKind((-1,0,0), B_sub_b,B_sub_b)] = 2*tA #b_05
    
    bulk[kwant.builder.HoppingKind((0,1,0), B_sub_c,B_sub_c)] = -2*tA #c_03
    bulk[kwant.builder.HoppingKind((0,-1,0), B_sub_c,B_sub_c)] = -2*tA #c_07
    bulk[kwant.builder.HoppingKind((1,0,0), B_sub_c,B_sub_c)] = -2*tA #c_01
    bulk[kwant.builder.HoppingKind((-1,0,0), B_sub_c,B_sub_c)] = -2*tA #c_05
	
    bulk[kwant.builder.HoppingKind((1,1,0), B_sub_a,B_sub_b)] = -1/2*tB #d_02
    bulk[kwant.builder.HoppingKind((-1,-1,0), B_sub_a,B_sub_b)] = -1/2*tB #d_06
    bulk[kwant.builder.HoppingKind((1,-1,0), B_sub_a,B_sub_b)] = 1/2*tB #d_08
    bulk[kwant.builder.HoppingKind((-1,1,0), B_sub_a,B_sub_b)] = 1/2*tB #d_04
     
    bulk[kwant.builder.HoppingKind((2,0,0), B_sub_a,B_sub_a)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((-2,0,0), B_sub_a,B_sub_a)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((0,2,0), B_sub_a,B_sub_a)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((0,-2,0), B_sub_a,B_sub_a)] = -3/2*tA 
    
    bulk[kwant.builder.HoppingKind((2,0,0), B_sub_b,B_sub_b)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((-2,0,0), B_sub_b,B_sub_b)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((0,2,0), B_sub_b,B_sub_b)] = -3/2*tA
    bulk[kwant.builder.HoppingKind((0,-2,0), B_sub_b,B_sub_b)] = -3/2*tA

    bulk[kwant.builder.HoppingKind((2,0,0), B_sub_c,B_sub_c)] = 3*tA
    bulk[kwant.builder.HoppingKind((-2,0,0), B_sub_c,B_sub_c)] = 3*tA
    bulk[kwant.builder.HoppingKind((0,2,0), B_sub_c,B_sub_c)] = 3*tA
    bulk[kwant.builder.HoppingKind((0,-2,0), B_sub_c,B_sub_c)] = 3*tA

    bulk[kwant.builder.HoppingKind((2,0,0), B_sub_a,B_sub_c)] = tC
    bulk[kwant.builder.HoppingKind((-2,0,0), B_sub_a,B_sub_c)] = -tC
    bulk[kwant.builder.HoppingKind((0,2,0), B_sub_b,B_sub_c)] = tC
    bulk[kwant.builder.HoppingKind((0,-2,0), B_sub_b,B_sub_c)] = -tC

    # Upper layer
	
    bulk[kwant.builder.HoppingKind((0,0,1), B_sub_a,B_sub_a)] = a_u0
    bulk[kwant.builder.HoppingKind((1,0,1), B_sub_a,B_sub_a)] = a_u1
    bulk[kwant.builder.HoppingKind((1,1,1), B_sub_a,B_sub_a)] = a_u2
    bulk[kwant.builder.HoppingKind((0,1,1), B_sub_a,B_sub_a)] =  a_u3
    bulk[kwant.builder.HoppingKind((-1,1,1), B_sub_a,B_sub_a)] = a_u4
    bulk[kwant.builder.HoppingKind((-1,0,1), B_sub_a,B_sub_a)] = a_u5
    bulk[kwant.builder.HoppingKind((-1,-1,1), B_sub_a,B_sub_a)] = a_u6
    bulk[kwant.builder.HoppingKind((0,-1,1), B_sub_a,B_sub_a)] = a_u7
    bulk[kwant.builder.HoppingKind((1,-1,1), B_sub_a,B_sub_a)] = a_u8
	
    bulk[kwant.builder.HoppingKind((0,0,1), B_sub_b,B_sub_b)] = b_u0
    bulk[kwant.builder.HoppingKind((1,0,1), B_sub_b,B_sub_b)] = b_u1
    bulk[kwant.builder.HoppingKind((1,1,1), B_sub_b,B_sub_b)] = b_u2
    bulk[kwant.builder.HoppingKind((0,1,1), B_sub_b,B_sub_b)] =  b_u3
    bulk[kwant.builder.HoppingKind((-1,1,1), B_sub_b,B_sub_b)] = b_u4
    bulk[kwant.builder.HoppingKind((-1,0,1), B_sub_b,B_sub_b)] = b_u5
    bulk[kwant.builder.HoppingKind((-1,-1,1), B_sub_b,B_sub_b)] = b_u6
    bulk[kwant.builder.HoppingKind((0,-1,1), B_sub_b,B_sub_b)] = b_u7
    bulk[kwant.builder.HoppingKind((1,-1,1), B_sub_b,B_sub_b)] = b_u8
	
    bulk[kwant.builder.HoppingKind((0,0,1), B_sub_c,B_sub_c)] = c_u0
    bulk[kwant.builder.HoppingKind((1,0,1), B_sub_c,B_sub_c)] = c_u1
    bulk[kwant.builder.HoppingKind((1,1,1), B_sub_c,B_sub_c)] = c_u2
    bulk[kwant.builder.HoppingKind((0,1,1), B_sub_c,B_sub_c)] =  c_u3
    bulk[kwant.builder.HoppingKind((-1,1,1), B_sub_c,B_sub_c)] = c_u4
    bulk[kwant.builder.HoppingKind((-1,0,1), B_sub_c,B_sub_c)] = c_u5
    bulk[kwant.builder.HoppingKind((-1,-1,1), B_sub_c,B_sub_c)] = c_u6
    bulk[kwant.builder.HoppingKind((0,-1,1), B_sub_c,B_sub_c)] = c_u7
    bulk[kwant.builder.HoppingKind((1,-1,1), B_sub_c,B_sub_c)] = c_u8
	
    bulk[kwant.builder.HoppingKind((0,0,1), B_sub_a,B_sub_b)] = d_u0
    bulk[kwant.builder.HoppingKind((1,0,1), B_sub_a,B_sub_b)] = d_u1
    bulk[kwant.builder.HoppingKind((1,1,1), B_sub_a,B_sub_b)] = d_u2
    bulk[kwant.builder.HoppingKind((0,1,1), B_sub_a,B_sub_b)] =  d_u3
    bulk[kwant.builder.HoppingKind((-1,1,1), B_sub_a,B_sub_b)] = d_u4
    bulk[kwant.builder.HoppingKind((-1,0,1), B_sub_a,B_sub_b)] = d_u5
    bulk[kwant.builder.HoppingKind((-1,-1,1), B_sub_a,B_sub_b)] = d_u6
    bulk[kwant.builder.HoppingKind((0,-1,1), B_sub_a,B_sub_b)] = d_u7
    bulk[kwant.builder.HoppingKind((1,-1,1), B_sub_a,B_sub_b)] = d_u8
	
    bulk[kwant.builder.HoppingKind((0,0,1), B_sub_a,B_sub_c)] = e_u0
    bulk[kwant.builder.HoppingKind((1,0,1), B_sub_a,B_sub_c)] = e_u1
    bulk[kwant.builder.HoppingKind((1,1,1), B_sub_a,B_sub_c)] = e_u2
    bulk[kwant.builder.HoppingKind((0,1,1), B_sub_a,B_sub_c)] =  e_u3
    bulk[kwant.builder.HoppingKind((-1,1,1), B_sub_a,B_sub_c)] = e_u4
    bulk[kwant.builder.HoppingKind((-1,0,1), B_sub_a,B_sub_c)] = e_u5
    bulk[kwant.builder.HoppingKind((-1,-1,1), B_sub_a,B_sub_c)] = e_u6
    bulk[kwant.builder.HoppingKind((0,-1,1), B_sub_a,B_sub_c)] = e_u7
    bulk[kwant.builder.HoppingKind((1,-1,1), B_sub_a,B_sub_c)] = e_u8
	
    bulk[kwant.builder.HoppingKind((0,0,1), B_sub_b,B_sub_c)] = f_u0
    bulk[kwant.builder.HoppingKind((1,0,1), B_sub_b,B_sub_c)] = f_u1
    bulk[kwant.builder.HoppingKind((1,1,1), B_sub_b,B_sub_c)] = f_u2
    bulk[kwant.builder.HoppingKind((0,1,1), B_sub_b,B_sub_c)] =  f_u3
    bulk[kwant.builder.HoppingKind((-1,1,1), B_sub_b,B_sub_c)] = f_u4
    bulk[kwant.builder.HoppingKind((-1,0,1), B_sub_b,B_sub_c)] = f_u5
    bulk[kwant.builder.HoppingKind((-1,-1,1), B_sub_b,B_sub_c)] = f_u6
    bulk[kwant.builder.HoppingKind((0,-1,1), B_sub_b,B_sub_c)] = f_u7
    bulk[kwant.builder.HoppingKind((1,-1,1), B_sub_b,B_sub_c)] = f_u8
	
    # Bottom layer
	
    bulk[kwant.builder.HoppingKind((0,0,-1), B_sub_a,B_sub_a)] = a_d0
    bulk[kwant.builder.HoppingKind((1,0,-1), B_sub_a,B_sub_a)] = a_d1
    bulk[kwant.builder.HoppingKind((1,1,-1), B_sub_a,B_sub_a)] = a_d2
    bulk[kwant.builder.HoppingKind((0,1,-1), B_sub_a,B_sub_a)] =  a_d3
    bulk[kwant.builder.HoppingKind((-1,1,-1), B_sub_a,B_sub_a)] = a_d4
    bulk[kwant.builder.HoppingKind((-1,0,-1), B_sub_a,B_sub_a)] = a_d5
    bulk[kwant.builder.HoppingKind((-1,-1,-1), B_sub_a,B_sub_a)] = a_d6
    bulk[kwant.builder.HoppingKind((0,-1,-1), B_sub_a,B_sub_a)] = a_d7
    bulk[kwant.builder.HoppingKind((1,-1,-1), B_sub_a,B_sub_a)] = a_d8
	
    bulk[kwant.builder.HoppingKind((0,0,-1), B_sub_b,B_sub_b)] = b_d0
    bulk[kwant.builder.HoppingKind((1,0,-1), B_sub_b,B_sub_b)] = b_d1
    bulk[kwant.builder.HoppingKind((1,1,-1), B_sub_b,B_sub_b)] = b_d2
    bulk[kwant.builder.HoppingKind((0,1,-1), B_sub_b,B_sub_b)] =  b_d3
    bulk[kwant.builder.HoppingKind((-1,1,-1), B_sub_b,B_sub_b)] = b_d4
    bulk[kwant.builder.HoppingKind((-1,0,-1), B_sub_b,B_sub_b)] = b_d5
    bulk[kwant.builder.HoppingKind((-1,-1,-1), B_sub_b,B_sub_b)] = b_d6
    bulk[kwant.builder.HoppingKind((0,-1,-1), B_sub_b,B_sub_b)] = b_d7
    bulk[kwant.builder.HoppingKind((1,-1,-1), B_sub_b,B_sub_b)] = b_d8
	
    bulk[kwant.builder.HoppingKind((0,0,-1), B_sub_c,B_sub_c)] = c_d0
    bulk[kwant.builder.HoppingKind((1,0,-1), B_sub_c,B_sub_c)] = c_d1
    bulk[kwant.builder.HoppingKind((1,1,-1), B_sub_c,B_sub_c)] = c_d2
    bulk[kwant.builder.HoppingKind((0,1,-1), B_sub_c,B_sub_c)] =  c_d3
    bulk[kwant.builder.HoppingKind((-1,1,-1), B_sub_c,B_sub_c)] = c_d4
    bulk[kwant.builder.HoppingKind((-1,0,-1), B_sub_c,B_sub_c)] = c_d5
    bulk[kwant.builder.HoppingKind((-1,-1,-1), B_sub_c,B_sub_c)] = c_d6
    bulk[kwant.builder.HoppingKind((0,-1,-1), B_sub_c,B_sub_c)] = c_d7
    bulk[kwant.builder.HoppingKind((1,-1,-1), B_sub_c,B_sub_c)] = c_d8
	
    bulk[kwant.builder.HoppingKind((0,0,-1), B_sub_a,B_sub_b)] = d_d0
    bulk[kwant.builder.HoppingKind((1,0,-1), B_sub_a,B_sub_b)] = d_d1
    bulk[kwant.builder.HoppingKind((1,1,-1), B_sub_a,B_sub_b)] = d_d2
    bulk[kwant.builder.HoppingKind((0,1,-1), B_sub_a,B_sub_b)] =  d_d3
    bulk[kwant.builder.HoppingKind((-1,1,-1), B_sub_a,B_sub_b)] = d_d4
    bulk[kwant.builder.HoppingKind((-1,0,-1), B_sub_a,B_sub_b)] = d_d5
    bulk[kwant.builder.HoppingKind((-1,-1,-1), B_sub_a,B_sub_b)] = d_d6
    bulk[kwant.builder.HoppingKind((0,-1,-1), B_sub_a,B_sub_b)] = d_d7
    bulk[kwant.builder.HoppingKind((1,-1,-1), B_sub_a,B_sub_b)] = d_d8
	
    bulk[kwant.builder.HoppingKind((0,0,-1), B_sub_a,B_sub_c)] = e_d0
    bulk[kwant.builder.HoppingKind((1,0,-1), B_sub_a,B_sub_c)] = e_d1
    bulk[kwant.builder.HoppingKind((1,1,-1), B_sub_a,B_sub_c)] = e_d2
    bulk[kwant.builder.HoppingKind((0,1,-1), B_sub_a,B_sub_c)] =  e_d3
    bulk[kwant.builder.HoppingKind((-1,1,-1), B_sub_a,B_sub_c)] = e_d4
    bulk[kwant.builder.HoppingKind((-1,0,-1), B_sub_a,B_sub_c)] = e_d5
    bulk[kwant.builder.HoppingKind((-1,-1,-1), B_sub_a,B_sub_c)] = e_d6
    bulk[kwant.builder.HoppingKind((0,-1,-1), B_sub_a,B_sub_c)] = e_d7
    bulk[kwant.builder.HoppingKind((1,-1,-1), B_sub_a,B_sub_c)] = e_d8
	
    bulk[kwant.builder.HoppingKind((0,0,-1), B_sub_b,B_sub_c)] = f_d0
    bulk[kwant.builder.HoppingKind((1,0,-1), B_sub_b,B_sub_c)] = f_d1
    bulk[kwant.builder.HoppingKind((1,1,-1), B_sub_b,B_sub_c)] = f_d2
    bulk[kwant.builder.HoppingKind((0,1,-1), B_sub_b,B_sub_c)] =  f_d3
    bulk[kwant.builder.HoppingKind((-1,1,-1), B_sub_b,B_sub_c)] = f_d4
    bulk[kwant.builder.HoppingKind((-1,0,-1), B_sub_b,B_sub_c)] = f_d5
    bulk[kwant.builder.HoppingKind((-1,-1,-1), B_sub_b,B_sub_c)] = f_d6
    bulk[kwant.builder.HoppingKind((0,-1,-1), B_sub_b,B_sub_c)] = f_d7
    bulk[kwant.builder.HoppingKind((1,-1,-1), B_sub_b,B_sub_c)] = f_d8
	
    return bulk

# Different geometries of the finite system

def trunc(site):
    x, y, z = abs(site.pos)
    return abs(x) <= 50 and abs(y) <= 50 and abs(z) <=1

def circle(site):
    x, y, z = site.pos
    r = 30
    return x ** 2 + y ** 2 < r ** 2

# Declare big things coming


for x in [0.00]:
    
    legend = []
    
    for d in [0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        
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
            model = make_system(type_embedded_C4 = 'multilayer')
            area_per_site = np.abs(lat_const*lat_const)
            syst.fill(model, trunc, (0, 0, 0));
            
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

        file_name = "Embedded_Square_DOS_dis_full_"+str(d)+"_mean_"+str(x)+"_L_"+str(L)+".dat"
        with open(file_name, "x") as file_fd:
            for i in range(0, len(est_energies)):
                file_fd.write(str(est_energies[i])+" "+str(est_densities[i])+" "+str(est_cond_xx_miu[i])+" "+str(est_cond_xy_miu[i])+"\n")
