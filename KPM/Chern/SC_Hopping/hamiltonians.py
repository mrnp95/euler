from itertools import product
import numpy as np
import kwant

from lcao import L_matrices, lcao_term


def standard_gaussian(label, salt, s1, s2=None):
    """Returns a gaussian distribution """
    return kwant.digest.gauss(str(hash((s1,s2))), salt=salt + label)


def uniform(input, label, salt):
    """Returns a uniform distribution in the interval '[-0.5, 0.5)'."""
    return kwant.digest.uniform(str(hash(str(input))), salt=salt + label) - 0.5

sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

sigmas = np.array([sigma_x, sigma_y, sigma_z])

chern_params = dict(m=0.5, m_std=0, t=1, t_std=0, salt='pepper')


def chern_insulator(L, pbc=False, repeat=0):
    """Make a Chern insulator model

    The system is a square sample of length 'L * (repeat + 1)'.
    The disorder configuration is repeated if 'repeated >= 1'.
    For example if 'repeated = 1' the disorder of the system is
       R R
       R R
    where 'R' represents a 'LxL' subsystem with a unique disorder
    configuration.

    Parameters
    ----------
    L : int
        Linear size of the (sub)system.
    pbc : bool, default to 'False
        If 'True', periodic boundary conditions are imposed on the
        hoppings of the system.
    repeat : int, default to '0'
        Number of copies of the disorder realization to append to each
        real space direction.
    """
    s_x = (sigma_z + 1j * sigma_x) / 2
    s_y = (sigma_z + 1j * sigma_y) / 2

    syst = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,0)),)

    lat = kwant.lattice.square(norbs=2)
    syst._lattice = lat

    total_length = L * (repeat + 1)
    l_map = np.array([1, L])

    def onsite(site, m, m_std, salt):
        this_order = uniform(l_map.dot(site.pos % L), "m", salt)
        return (m + m_std * this_order) * sigma_z

    for x in range(total_length):
        for y in range(total_length):
            syst[lat(x, y)] = onsite

    def hop_x(site1, site2, t, t_std, salt):
        this_order = uniform(l_map.dot(site1.pos % L), "tx", salt)
        return (t + t_std * this_order) * s_x

    def hop_y(site1, site2, t, t_std, salt):
        this_order = uniform(l_map.dot(site1.pos % L), "ty", salt)
        return (t + t_std * this_order) * s_y

    for x, y in product(range(total_length - 1 + 1 * pbc), repeat=2):
        xp = (x + 1) % total_length
        yp = (y + 1) % total_length
        syst[(lat(x, y), lat(xp, y))] = hop_x
        syst[(lat(x, y), lat(x, yp))] = hop_y

    return syst.finalized()


# Parameters listed for the two sublattices
SnTe_6band_params = dict(
    t = 2 * np.array([[-0.25, 0.45], [0.45, 0.25]]), # hoppings
    m = np.array([-1.65, 1.65]), # onsite energies
    lam = np.array([-0.3, -0.3]), # onsite spin-orbit couplings
    )


def SnTe_6band(translations=None):
    """Make bulk 6-band model of SnTe from 
    `<https://arxiv.org/pdf/1202.1003.pdf>`_

    Parameters
    ----------
    translations: kwant.lattice.TranslationalSymmetry
        Optional different translational unit cell.
        Should still be a primitive UC.
    """

    L = L_matrices()
    L_dot_S = np.sum([np.kron(sigmas[i], L[i]) for i in range(3)], axis=0)

    def onsite(site, m, lam, mu):
        # which sublattice
        a = np.sum(site.tag) % 2
        # Use position dependent mass
        os = (mu + m(site)) * np.eye(6)
        # L dot S onsite SoC
        spinorb = lam[a] * L_dot_S
        # x, y, z = site.pos
        os = os + spinorb
        return os

    def hopping(site1, site2, t):
        # which sublattice
        a = np.sum(site1.tag) % 2
        b = np.sum(site2.tag) % 2
        d = site1.tag - site2.tag
        # ppsigma bonding
        dtd = np.kron(np.eye(2), lcao_term(1, 1, 0, d))
        # Use the appropriate hopping depending on sublattices
        hop = t[a, b] * dtd
        return hop

    # Cubic rocksalt structure with FCC symmetry
    lat = kwant.lattice.general(np.eye(3), norbs=6)
    if translations is None:
        # Default translation vectors of FCC structure
        translations = kwant.lattice.TranslationalSymmetry(
            [1, 1, 0], [1, 0, 1], [0, 1, 1])
    syst = kwant.Builder(symmetry=translations)

    # Populate the builder using the cubic lattice sites
    # Two sublattices
    syst[lat(0, 0, 0)] = onsite
    syst[lat(0, 0, 1)] = onsite

    # First and second neighbor hoppings
    syst[lat.neighbors(1)] = hopping
    syst[lat.neighbors(2)] = hopping

    return syst


def SnTe_6band_disorder(translations=None):
    """Make bulk 6-band model of SnTe from https://arxiv.org/pdf/1202.1003.pdf

    translations: kwant.lattice.TranslationalSymmetry
        Optional different translational unit cell. Should still be a primitive UC.
    """

    L = L_matrices()
    L_dot_S = np.sum([np.kron(sigmas[i], L[i]) for i in range(3)], axis=0)

    def onsite(site, m, lam, mu):
        # which sublattice
        a = np.sum(site.tag) % 2
        # Use position dependent mass
        os = (mu + m(site)) * np.eye(6)
        # L dot S onsite SoC
        spinorb = lam[a] * L_dot_S
        # x, y, z = site.pos
        os = os + spinorb
        return os

    def hopping(site1, site2, t):
        # which sublattice
        a = np.sum(site1.tag) % 2
        b = np.sum(site2.tag) % 2
        d = site1.tag - site2.tag
        # ppsigma bonding
        dtd = np.kron(np.eye(2), lcao_term(1, 1, 0, d))
        # Use the appropriate hopping depending on sublattices
        hop = t[a, b] * dtd
        return hop

    # Cubic rocksalt structure with FCC symmetry
    lat = kwant.lattice.general(np.eye(3), norbs=6)
    if translations is None:
        # Default translation vectors of FCC structure
        translations = kwant.lattice.TranslationalSymmetry([1, 1, 0], [1, 0, 1], [0, 1, 1])
    syst = kwant.Builder(symmetry=translations)

    # Populate the builder using the cubic lattice sites
    # Two sublattices
    syst[lat(0, 0, 0)] = onsite
    syst[lat(0, 0, 1)] = onsite

    # First and second neighbor hoppings
    syst[lat.neighbors(1)] = hopping
    syst[lat.neighbors(2)] = hopping

    return syst


def doped_m(site, doping, mSn, mPb, mTe, n=np.array([1, 1, 0]), to_fd=None, salt='salt'):
    tag = site.tag
    a = np.sum(tag) % 2
    if a == 0:
        return mTe
    else:
        # Make it mirror symmetric
        if n is not None:
            Mtag = tag - (2 * np.dot(n, tag) * n) // np.dot(n, n)
            if to_fd is not None:
                Mtag = to_fd(site.family(*Mtag)).tag
            tag = max(tag, Mtag)
        ran = kwant.digest.uniform(tag, salt=salt)
        if ran < doping:
            return mPb
        else:
            return mSn


# Define 18-orbital model including spinful s, p, d
# orbitals from Lent et.al. Superlattices and
# Microstructures, Vol. 2, Issue 5, 491-499, (1986).
# Sign choice is consistent with extra minus signs in vps and vsp
# this is only a gauge transformation adding - sign
# to all s orbitals
SnTe_18band_params = dict(
            esc= -6.578,
            esa= -12.067,
            epc= 1.659,
            epa= -0.167,
            edc= 8.38,
            eda= 7.73,
            lambdac= 0.592,
            lambdaa= 0.564,
            vss= -0.510,
            vsp= -1*0.949,
            vps= -1*-0.198,
            vpp= 2.218,
            vpppi= -0.446,
            vpd= -1.11,
            vpdpi= 0.624,
            vdp= -1.67,
            vdppi= 0.766,
            vdd= -1.72,
            vdddelta= 0.618,
            vddpi=0,
            )


PbTe_18band_params = dict(
            esc= -7.612,
            esa= -11.002,
            epc= 3.195,
            epa= -0.237,
            edc= 7.73,
            eda= 7.73,
            lambdac= 1.500,
            lambdaa= 0.428,
            vss= -0.474,
            vsp= -1*0.705,
            vps= -1*0.633,
            vpp= 2.066,
            vpppi= -0.430,
            vpd= -1.29,
            vpdpi= 0.835,
            vdp= -1.59,
            vdppi= 0.531,
            vdd= -1.35,
            vdddelta= 0.668,
            vddpi=0,
            )


def SnTe_18band_disorder(translations=None):
    """Make disordered 18-band model of SnTe from Lent et.al.
    Superlattices and Microstructures, Vol. 2, Issue 5, 491-499, (1986).

    translations: kwant.lattice.TranslationalSymmetry
        Optional different translational unit cell. Should still be a primitive UC.
    """
    L = L_matrices()
    L_dot_S = np.sum([np.kron(L[i], 0.5 * sigmas[i]) for i in range(3)], axis=0)

    @ft.lru_cache(100)
    def H_os(es, ep, ed, lam):
        H = np.zeros((18, 18), dtype=complex)
        H[:2, :2] = es * np.eye(2)
        H[2:8, 2:8] = ep * np.eye(6) + lam * L_dot_S
        H[8:18, 8:18] = ed * np.eye(10)
        return H

    @ft.lru_cache(100)
    def H_ac(d, vss, vsp, vps, vpp, vpppi, vpd, vpdpi,
             vdp, vdppi, vdd, vdddelta, vddpi):
        d = ta.array(d)
        Hac = np.zeros((18, 18), dtype=complex)
        Hac[:2, :2] = vss * np.kron(lcao_term(0, 0, 0, d), np.eye(2))
        Hac[2:8, :2] = vsp * np.kron(lcao_term(1, 0, 0, d), np.eye(2))
        Hac[:2, 2:8] = vps * np.kron(lcao_term(0, 1, 0, d), np.eye(2))
        Hac[2:8, 2:8] = (vpp * np.kron(lcao_term(1, 1, 0, d), np.eye(2))
                        + vpppi * np.kron(lcao_term(1, 1, 1, d), np.eye(2)))
        Hac[8:18, 2:8] = (vpd * np.kron(lcao_term(2, 1, 0, d), np.eye(2))
                         + vpdpi * np.kron(lcao_term(2, 1, 1, d), np.eye(2)))
        Hac[2:8, 8:18] = (vdp * np.kron(lcao_term(1, 2, 0, d), np.eye(2))
                         + vdppi * np.kron(lcao_term(1, 2, 1, d), np.eye(2)))
        Hac[8:18, 8:18] = (vdd * np.kron(lcao_term(2, 2, 0, d), np.eye(2))
                          + vddpi * np.kron(lcao_term(2, 2, 1, d), np.eye(2))
                          + vdddelta * np.kron(lcao_term(2, 2, 2, d), np.eye(2)))
        return Hac


    def onsite(site, esa, epa, eda, lambdaa, esc, epc, edc, lambdac, site_type):
        if np.sum(site.tag) % 2 == 0:
            # Keep the first material's onsite on a sites
            x = site_type(site)
            return H_os(x @ esa, x @ epa, x @ eda, x @ lambdaa)
        else:
            # Find out type of site, only dope c sites
            x = site_type(site)
            return H_os(x @ esc, x @ epc, x @ edc, x @ lambdac)

    def hopping(site1, site2, vss, vsp, vps, vpp, vpppi, vpd, vpdpi,
             vdp, vdppi, vdd, vdddelta, vddpi, site_type):
        # convert to tinyarray for caching
        d = ta.array(site2.pos - site1.pos)
        if np.allclose(d, np.round(d)):
            d = ta.array(np.round(d), int)
        if np.isclose(np.sum(site1.tag) % 2, 0):
            # Find out type of site, only dope c sites
            x = site_type(site2)
            d = d
            conj = False
        else:
            x = site_type(site1)
            d = -d
            conj = True
        hop = H_ac(-d, x @ vss, x @ vsp, x @ vps, x @ vpp, x @ vpppi, x @ vpd, x @ vpdpi,
                        x @ vdp, x @ vdppi, x @ vdd, x @ vdddelta, x @ vddpi)
        if conj:
            hop = hop.T.conj()
        return hop

    # Cubic rocksalt structure with FCC symmetry
    lat = kwant.lattice.general(np.eye(3), norbs=18)
    if translations is None:
        # Default translation vectors of FCC structure
        translations = kwant.lattice.TranslationalSymmetry([1, 1, 0], [1, 0, 1], [0, 1, 1])
    syst = kwant.Builder(symmetry=translations)

    # Populate the builder using the cubic lattice sites
    # Two sublattices
    syst[lat(0, 0, 0)] = onsite
    syst[lat(0, 0, 1)] = onsite

    # First neighbor hoppings
    syst[lat.neighbors(1)] = hopping

    return syst


def site_type(site, doping, n=np.array([1, 1, 0]), to_fd=None, salt='salt'):
    # Return a vector that specifies the mixture of the two materials'
    # parameters on the site.
    if to_fd is not None:
        site = to_fd(site)
    tag = site.tag
    a = np.sum(tag) % 2
    if a == 0:
        # For a sites take the average type of the neighbors
        neighbors = [site.family(*(site.tag + d)) for d in np.concatenate([np.eye(3), -np.eye(3)])]
        return np.sum([site_type(site_n, doping, n, to_fd, salt) for site_n in neighbors], axis=0) / 6
    else:
        if n is not None:
            # Make it mirror symmetric if n is not None
            Mtag = tag - (2 * np.dot(n, tag) * n) // np.dot(n, n)
            if to_fd is not None:
                Mtag = to_fd(site.family(*Mtag)).tag
            tag = max(tag, Mtag)
        ran = kwant.digest.uniform(tag, salt=salt)
        if ran < doping:
            # Dopant, purely the second material
            return np.array([0, 1])
        else:
            # Original, purely the first material
            return np.array([1, 0])


SnPbTe_params = {key: np.array([val, PbTe_18band_params[key]]) for key, val in SnTe_18band_params.items()}
