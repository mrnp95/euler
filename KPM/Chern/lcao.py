import numpy as np
from scipy.linalg import norm
import functools as ft

# Generate Hamiltonian terms corresponding to orbital bonding in the LCAO picture.
# See https://journals.aps.org/pr/abstract/10.1103/PhysRev.94.1498
# Orbitals are given in the real (cubic harmonic) basis:
# [s]
# [p_x, p_y, p_z]
# [d_{x^2 - y^2}, d_{3 z^2 - r^2}, d_{xy}, d_{yz}, d_{zx}]

@ft.lru_cache(100)
def lcao_term(l1, l2, md, d12):
    """Return the LCAO matrix between orbitals with total angular
    momentum `l1` and `l2` on sites separated by `d12` with angular
    momentum absolute value `m` along the bonding axis.

    Parameters:
    -----------
    l1, l2 : int
        Total angular momentum of states on site1 and site2. Usually
        denoted [s, p, d, f, ...] for [0, 1, 2, 3, ...].
    md : int
        Absolute value of angular momentum along the bonding axis.
        Usually denoted [sigma, pi, delta, phi, ...] for [0, 1, 2, 3, ...].
    d12 : arraylike
        3 component real vector, bond vector in real space.

    Returns:
    --------
    H : ndarray
        Hamiltonian term, matrix of shape (2 * l1 + 1, 2 * l2 + 1)
    """

    d12 = np.array(d12) / norm(d12)
    l, m, n = d12
    s3 = np.sqrt(3)

    if md > min(l1, l2):
        return np.zeros((2 * l1 + 1, 2 * l2 + 1), dtype=complex)

    if max(l1, l2) > 2:
        raise NotImlpementedError['f and higher orbitals not implemented']

    if md == 0:
        # sigma terms
        # m=0 states for s, p, d
        psi0 = [
            [1], d12, [s3 / 2 * (l**2 - m**2),
                       n**2 - (l**2 + m**2) / 2,
                       s3 * l * m,
                       s3 * m * n,
                       s3 * n * l]]
        return np.outer(psi0[l1], psi0[l2])

    elif l1 == 1 and l2 == 1 and md == 1:
        # p, p, pi term
        return np.eye(3) - np.outer(d12, d12)

    elif (l1 == 1 and l2 == 2) or (l1 == 2 and l2 == 1):
        # p, d, pi or d, p, pi terms
        H = np.array([
            [l * (1 - l**2 + m**2),
             -s3 * l * n**2, m * (1 - 2 * l**2),
             -2 * l * m * n,
             n * (1 - 2 * l**2)],
            [-m * (1 + l**2 - m**2),
             -s3 * m * n**2,
             l * (1 - 2 * m**2),
             n * (1 - 2 * m**2),
             -2 * l * m * n],
            [-n * (l**2 - m**2),
             -s3 * n * (l**2 + m**2),
             -2 * l * m * n,
             m * (1 - 2 * n**2),
             l * (1 - 2 * n**2)]
            ])
        return (H if l1 == 1 else H.T)

    elif l1 == 2 and l2 == 2:
        # Angular momentum along bonding axis
        L_dxyz = L_matrices(l=2)
        Ld = np.tensordot(L_dxyz, d12, (0, 0))
        Ld2 = Ld.dot(Ld)
        Ld4 = Ld2.dot(Ld2)
        # Make projector onto +/- m subspace
        if md == 1:
            # d, d, pi term
            return 1/3 * (4 * Ld2 - Ld4)
        elif md == 2:
            # d, d, delta term
            return 1/12 * (Ld4 - Ld2)

    else:
        raise NotImlpementedError['f and higher orbitals not implemented']


def L_matrices(d=3, l=1):
    """Construct real space rotation generator matrices in d=2 or 3 dimensions.

    Can also be used to get angular momentum operators for real atomic orbitals
    in 3 dimensions, for p-orbitals use `l=1`, for d-orbitals `l=2`. The basis
    of p-orbitals is `p_x`, `p_y`, `p_z`, for d-orbitals `d_{x^2 - y^2}`,
    `d_{3 z^2 - r^2}`, `d_{xy}`, `d_{yz}`, `d_{zx}`. The matrices are all
    purely imaginary and antisymmetric.
    """
    if d == 2 and l==1:
        return 1j * np.array([[[0, -1], [1, 0]]])
    elif d == 3 and l==1:
        return 1j * np.array([[[ 0,  0,  0],
                               [ 0,  0, -1],
                               [ 0,  1,  0]],
                              [[ 0,  0,  1],
                               [ 0,  0,  0],
                               [-1,  0,  0]],
                              [[ 0, -1,  0],
                               [ 1,  0,  0],
                               [ 0,  0,  0]]])
    elif d == 3 and l==2:
        s3 = np.sqrt(3)
        return 1j * np.array([
            [[0,   0,  0,  -1,  0],
             [0,   0,  0, -s3,  0],
             [0,   0,  0,   0,  1],
             [1,  s3,  0,   0,  0],
             [0,   0, -1,   0,  0]],
            [[0,   0,  0,   0, -1],
             [0,   0,  0,   0, s3],
             [0,   0,  0,  -1,  0],
             [0,   0,  1,   0,  0],
             [1, -s3,  0,   0,  0]],
            [[0,   0, -2,   0,  0],
             [0,   0,  0,   0,  0],
             [2,   0,  0,   0,  0],
             [0,   0,  0,   0,  1],
             [0,   0,  0,  -1,  0]]
            ])
    else:
        raise ValueError('Only 2 and 3 dimensions are supported.')
