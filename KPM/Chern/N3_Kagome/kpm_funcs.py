import scipy
from scipy.sparse import diags
import numpy as np

import kwant
from kwant.kpm import jackson_kernel

def position_operator(syst, params=None, pos_transform=None):
    """Return a list of position operators."""
    operators = []
    norbs = syst.sites[0].family.norbs
    if pos_transform is None:
        pos = np.array([s.pos for s in syst.sites])
    else:
        pos = np.array([pos_transform(s.pos) for s in syst.sites])
    for c in range(pos.shape[1]):
        operators.append(diags(np.repeat(pos[:, c], norbs), format='csr'))
    return operators


def build_projector(ham, vectors, kpm_params=dict(), params=None):
    """Build a projector over the occupied energies.

    Returns a function that takes a Fermi energy, and returns the
    projection of the `vectors` over the occupied energies of the
    Hamiltonian.

    Parameters
    ----------
    ham : kwant.System or matrix
        Finalized kwant system or matrix Hamiltonian.
    vectors : iterable of vectors
        Vectors upon which the projector will act. Must comply with the
        requirements of 'vector_factory' in `~kwant.kpm.SpectralDensity`.
    kpm_params : dict, optional
        Dictionary containing the parameters to pass to the
        `~kwant.kpm.SpectralDensity` module. 'num_moments' is the order
        of the expansion, 'num_vectors', 'operator', and 'vector_factory'
        will be overwritten, if present.
    params : dict, optional
        Parameters for the kwant system.
    """
    # set to 100 if absent
    num_moments = kpm_params.get('num_moments', 100)
    kpm_params['num_moments'] = num_moments
    # set to None if absent
    kpm_params['num_vectors'] = kpm_params.get('num_vectors', None)
    kpm_params['vector_factory'] = vectors
    kpm_params['operator'] = lambda bra, ket: ket

    spectrum = kwant.kpm.SpectralDensity(ham, params=params, **kpm_params)
    expanded_vectors = np.asarray(spectrum._moments_list)
    def projected_vectors(e=0):
        """Return the vectors projected to energies below 'e' """
        # scaled energies
        e = (np.atleast_1d(e) - spectrum._b) / spectrum._a
        phi = np.arccos(e)
        gs = jackson_kernel(np.ones(num_moments))
        m = np.arange(1, num_moments)
        coef = -gs[1:] * np.sin(np.outer(m, phi)).T / m
        coef = np.concatenate([0.5 * (np.pi - phi)[:, None], coef], axis=1)
        return 2 / np.pi * coef @ expanded_vectors

    return projected_vectors


def kpm_vector_generator(ham, vectors, max_moments):
    """
    Generator object that succesively yields KPM vectors `T_n(ham) |vector>`
    for vectors in `vectors`.

    Parameters
    ----------
    vectors : 1D or 2D array
        Vector or set of column vectors on which to apply the projector `Theeta(e_f - H)`.
    ham : 2D array
        Hamiltonian
    max_moments : int
        Number of moments to stop with iteration
    """
    alpha_prev = np.zeros(vectors.shape, dtype=complex)
    alpha = np.zeros(vectors.shape, dtype=complex)
    alpha_next = np.zeros(vectors.shape, dtype=complex)

    alpha[:] = vectors
    n = 0
    yield alpha
    n += 1
    alpha_prev[:] = alpha
    alpha[:] = ham @ alpha
    yield alpha
    n += 1
    while n < max_moments:
        alpha_next[:] = 2 * ham @ alpha - alpha_prev
        alpha_prev[:] = alpha
        alpha[:] = alpha_next
        yield alpha
        n += 1


def projector(ham, vectors, kpm_params, e_F=0, bounds=None):
    """Projector over the filled states

    Projects a vector into the subspace of eigenvectors with eigenvalues
    below the Fermi level `e_F`.
    Basically is the polynomial expansion of `Theeta(e_f - H)`,
    where `H` is the `syst` Hamiltonian and `Theeta()` the Heaviside
    function.

    Parameters
    ----------
    vectors : numpy ndarray
        Vector or set of column vectors on which to apply the projector `Theeta(e_f - H)`.
    ham : numpy ndarray
        Hamiltonian to pass to the KPM module.
    e_F : scalar, default '0'
        Fermi level of the system.
    bounds : tuple or None
        (lmin, lmax) bounds of the spectrum. If None, the bounds
        are calculated.
    """
    num_moments = kpm_params.get('num_moments', 100)
    ham = scipy.sparse.csr_matrix(ham)
    ham_rescaled, (a, b) = kwant.kpm._rescale(ham, 0.05, None, bounds)

    phi_f = np.arccos((e_F - b) / a)
    gs = jackson_kernel(np.ones(num_moments))
    gs[0] = 0
    m = np.arange(num_moments)
    m[0] = 1
    coef = np.multiply(gs, (np.divide(np.sin(m * phi_f), np.pi * m)))
    coef_times_vectors = sum(c * vec for c, vec
                            in zip(coef, kpm_vector_generator(ham_rescaled, vectors, num_moments)))
    # print(coef_times_vectors.shape)
    return (1 - phi_f/np.pi) * vectors - 2 * coef_times_vectors
