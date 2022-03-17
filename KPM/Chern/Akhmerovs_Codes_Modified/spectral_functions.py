import numpy as np
import scipy.linalg as la
import itertools as it
import kwant


def surface_dos(slab, k_p, params, supercell=np.eye(3), pos_transform=None, num_moments=200,
                      bounds=None, window=None):
    """Calculate k-dependent spectral function for system slab with two translation
    invariant directions."""
    if pos_transform is None:
        pos_transform = lambda x: x
    supercell = np.array(supercell)
    params = params.copy()
    G = la.inv(supercell).T
    if window is None:
        window = lambda pos: pos @ G[2] > 0.5 - la.norm(G[2])

    # assume same norbs on every site
    norbs = slab.sites[0].family.norbs
    v_ks = np.empty((norbs, 0), dtype=complex)
    for site in slab.sites:
        pos = pos_transform(site.pos)
        if window(pos):
            v_ks = np.hstack([v_ks, np.exp(1j * np.dot(pos, k_p)) * np.eye(norbs)])
        else:
            v_ks = np.hstack([v_ks, 0 * np.eye(norbs)])

    vf = iter(v_ks)

    params['k_x'] = k_p @ np.array(slab._wrapped_symmetry.periods)[0]
    params['k_y'] = k_p @ np.array(slab._wrapped_symmetry.periods)[1]

    spectrum = kwant.kpm.SpectralDensity(slab,
                                         params=params,
                                         vector_factory=vf,
                                         num_moments=num_moments,
                                         num_vectors=norbs,
                                         bounds=bounds)
    return spectrum
