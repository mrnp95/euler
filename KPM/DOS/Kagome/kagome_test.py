import kwant
import matplotlib.pyplot as plt

def circle(pos):
    x, y = pos
    return x**2 + y**2 < 5000

lat = kwant.lattice.kagome()
syst = kwant.Builder()
syst[lat.shape(circle, (0, 0))] = 0
syst[lat.neighbors()] = -1.0

fsyst = syst.finalized()
rho = kwant.kpm.SpectralDensity(fsyst, num_moments = 1000)

energies, densities = rho.energies, rho.densities
plt.plot(energies, densities)
plt.xlim(-3, 1)
plt.ylim(0, 100000)
plt.show()
