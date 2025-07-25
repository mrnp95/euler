import functools as ft
import collections
import scipy
import scipy.sparse
import scipy.linalg as la
import numpy as np
import kwant
from kpm_funcs import projector, position_operator
from lcao import L_matrices
from hamiltonians import sigmas


def mirror_chern(syst, x, y, vectors, Mz=None, e_F=0, window=None,
                 kpm_params=dict(), params=None, bounds=None, return_std=False):
    """(Mirror) Chern number averaged over `vectors`.
    Defined as `Tr Mz (P X P Y P - P Y P X P)`, where
    `P` is the projector over the occupied states below `e_F`,
    `X` and `Y` are the position operators. Tr is evaluated
    with `vectors`.

    Parameters
    ----------
    syst : kwant system or sparse matrix
        Quasi three-dimensional finalized system or the corresponding
        hamiltonian_submatrix.
    x, y : sparse matrices
        Position operators in the basis of the Hamiltonian.
    Mz : sparse matrix or None
        Mirror operator in the basis of the Hamiltonian.
        If None, the regular Chern number is calculated.
    e_F : float
        Fermi energy.
    params : dict
        Parameters to pass to the system.
    kpm_params = dict, optional
        Parameters for the kpm expansion of the projector operator.
    vectors : int or iterable of vectors
        If integer, number of random phase vectors. In this case
        `window` must be provided.
        Vectors must an iterable of vectors of length N where N is the total number of
        orbitals in the system.
    window : callable or sparse matrix or None
        Window defining the averaging region, only used if `vectors` is int.
        If callable, must take a site and return a boolean. Only works
        if `syst` is kwant system.
        If sparse matrix, must be the projector onto the window in the
        basis of the Hamiltonian.
    return_std : boolean
        Whether to return the standard deviation of the Chern number
        or the list of Chern numbers.

    Returns
    -------
    nu : float
        Estimated total Chern number. Has to be divided by the area
        that `window` or `vectors` cover to get the invariant.
    nu_std : float
        Returned only if return_std=True. Standard deviation of nu
        when calculated for different random vectors.
    """

    if isinstance(syst, kwant.system.System):
        ham = syst.hamiltonian_submatrix(params=params, sparse=True).tocsr()
    else:
        ham = syst

    if callable(window):
        window = make_window(syst, window)

    if isinstance(vectors, collections.Iterable):
        # do nothing if iterable
        pass
    elif isinstance(vectors, int):
        vectors = random_phase_vecs(vectors, ham.shape[0], window)
    else:
        raise ValueError("'vectors' must be an `int` or `iterable`.")

    # Make Mz square to 1
    Id = scipy.sparse.eye(ham.shape[0], format="csr")
    if Mz is None:
        # If Mz is not provided, use identity
        Mz = Id
    else:
        Mz2 = Mz @ Mz
        if np.allclose((Mz2 + Id).data, 0):
            Mz = 1j * Mz
        elif not np.allclose((Mz2 - Id).data, 0):
            raise ValueError('Mz must square to +/-1')

    # we will iterate one vector at the time to save memory
    p = lambda vec: projector(vectors=vec, kpm_params=kpm_params,
                              ham=ham, e_F=e_F, bounds=bounds)

    Cs = []
    # Average over num_vecs random vectors
    for vector in vectors:
        p_vector = p(vector)
        pxp_vector = p(x @ p_vector)
        yp_vector = y @ p_vector
        # This formula assumes that the Hamiltonian (also the projector)
        # is invariant under Mz and Mz has +/-1 eigenvalues.
        C = yp_vector.T.conjugate() @ (Mz @ pxp_vector)
        C = -4 * np.pi * C.imag
        Cs.append(C)

    if return_std:
        return np.mean(Cs), np.std(Cs)
    else:
        return Cs


def make_window(syst, window):
    window_op = kwant.operator.Density(syst,
                                       onsite=lambda site: np.eye(site.family.norbs),
                                       where=window)
    window_op = window_op.bind(params=dict())
    window_op = window_op.tocoo()
    window_op.eliminate_zeros()
    return window_op


def random_phase_vecs(num_vecs, ham_size, window=None):
    for _ in range(num_vecs):
        vec = np.exp(2j * np.pi * np.random.random(ham_size))
        if window is None:
            yield vec
        else:
            yield window @ vec


def orbital_slices(syst):
    orbital_slices = {}
    start_orb = 0

    for site in syst.sites:
        n = site.family.norbs
        orbital_slices[site] = slice(start_orb, start_orb + n)
        start_orb += n
    return orbital_slices


def pg_op(syst, trf, U):
    '''
    Build SG operator for finite system.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        Finite system, generated by wraparound.
        Cannot have any translational symmetry and its lattices
        must have n_orbs set.
    trf : callable
        Spatial transformation of the group element. Must take a
        site and return the transformed site.
    U : ndarray or dict: lattice -> ndarray
        Unitary onsite action of spacegroup operator. If a single
        square matrix is provided it will be used for all lattices,
        to use different unitaries, provide in a dict for every lattice.

    Returns
    -------
    op : ndarray
        Space group operation in the basis of hamiltonian_submatrix
        as scipy.sparse.csr_matrix.
    '''
    if isinstance(syst.symmetry, kwant.lattice.TranslationalSymmetry):
        raise ValueError('syst must not have any translational symmetry')
    # Make it work with PBC through wraparound
    try:
        to_fd = syst._wrapped_symmetry.to_fd
    except:
        to_fd = lambda site: site

    _, _, N = syst.site_ranges[-1]
    op = scipy.sparse.lil_matrix((N, N), dtype=complex)
    slices = orbital_slices(syst)

    for site in syst.sites:
        lat = site.family
        if isinstance(U, dict):
            Ul = U[lat]
        else:
            Ul = U
        trf_site = to_fd(trf(site))
        op[slices[trf_site], slices[site]] = Ul
    return op.tocsr()


def M_cubic(site, n):
    lat = site.family
    tag = site.tag
    # Should always be integer
    new_tag = tag - (2 * np.dot(n, tag) * n) / np.dot(n, n)
    new_tag_int = np.array(new_tag, dtype=int)
    assert np.allclose(new_tag, new_tag_int)
    return lat(*new_tag_int)


def UM_p(n):
    # Define angular momentum matrices for spinful p-orbitals
    L = L_matrices()
    J = [np.kron(np.eye(2), L[i]) + np.kron(1/2 * sigmas[i], np.eye(3)) for i in range(3)]
    # Define inversion operator (p-orbitals)
    Inv = np.kron(np.eye(2), -np.eye(3))

    # Mirror operator wrt n
    n_norm = n/la.norm(n) * np.pi
    UM = la.expm(-1j * np.tensordot(n_norm, J , axes=((0), (0)))) @ Inv
    # get rid of small entries
    UM[abs(UM) < 1e-15] = 0
    # should square to -1
    assert np.allclose(UM @ UM, -np.eye(6), atol=1e-15)
    return UM


def UM_spd(n):
    # Define angular momentum matrices for spinful spd-orbitals
    Lp = L_matrices(l=1)
    Ld = L_matrices(l=2)
    L = [scipy.linalg.block_diag(0*np.eye(1), Lpi, Ldi) for Lpi, Ldi in zip(Lp, Ld)]
    J = np.array([np.kron(L[i], np.eye(2), ) + np.kron(np.eye(9), 1/2 * sigmas[i]) for i in range(3)])
    # Define inversion operator (p-orbitals)
    Inv = np.kron(scipy.linalg.block_diag(np.eye(1), -np.eye(3), np.eye(5)), np.eye(2))

    # Mirror operator wrt n
    n_norm = n/la.norm(n) * np.pi
    UM = la.expm(-1j * np.tensordot(n_norm, J, axes=((0), (0)))) @ Inv
    # get rid of small entries
    UM[abs(UM) < 1e-15] = 0
    # should square to -1
    assert np.allclose(UM @ UM, -np.eye(18), atol=1e-15), np.diag(UM @ UM)
    return UM

