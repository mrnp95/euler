import kwant
import numpy as np
import matplotlib.pyplot as plt

def make_syst_topo(r=30, a=1, t=1, t2=0.5):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1, name=['a', 'b'])

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.shape(circle, (0, 0))] = 0.
    syst[lat.neighbors()] = t
    # add second neighbours hoppings
    syst[lat.a.neighbors()] = 1j * t2
    syst[lat.b.neighbors()] = -1j * t2
    syst.eradicate_dangling()

    return lat, syst.finalized()

# construct the Haldane model
lat, fsyst_topo = make_syst_topo()
# find 'A' and 'B' sites in the unit cell at the center of the disk
where = lambda s: np.linalg.norm(s.pos) < 1

# component 'xx'
s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)
cond_xx = kwant.kpm.conductivity(fsyst_topo, alpha='x', beta='x', mean=True,
                                         num_vectors=None, vector_factory=s_factory,
                                                                          rng=0)
# component 'xy'
s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)
cond_xy = kwant.kpm.conductivity(fsyst_topo, alpha='x', beta='y', mean=True,
                                         num_vectors=None, vector_factory=s_factory,
                                                                          rng=0)

energies = cond_xx.energies
cond_array_xx = np.array([cond_xx(e, temperature=0.01) for e in energies])
cond_array_xy = np.array([cond_xy(e, temperature=0.01) for e in energies])

# area of the unit cell per site
area_per_site = np.abs(np.cross(*lat.prim_vecs)) / len(lat.sublattices)
cond_array_xx /= area_per_site
cond_array_xy /= area_per_site

plt.plot(energies, cond_array_xx/4)
plt.plot(energies, cond_array_xy)
plt.legend(["$\sigma_{xx}/4$","$\sigma_{xy}$"])
plt.title("Haldane conductivities")
plt.show()

