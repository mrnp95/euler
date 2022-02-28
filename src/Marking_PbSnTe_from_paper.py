# Make a slab with PBC in one direction and open BC in the orthogonal directions
# surface normal
n = np.array([1, 1, 0])
n11 = np.array([1, -1, 0])
nz = np.array([0, 0, 1])
# thickness (number of atomic layers - 1)
W = 40
L11 = 40
Lz = 60

num_vectors = 5
num_moments = 1000
# salt specifies the disorder realization used
salt = '2'

num_m = 51
m_array = np.linspace(-1, 4, num_m)

num_x = 21
x_array = np.linspace(0, 1, num_x)

def make_operators(doping, mPb):
    syst2 = SnTe_6band_disorder()

    # Build film using syst
    film = kwant.Builder(kwant.lattice.TranslationalSymmetry(W * n, L11 * n11, Lz * nz))

    film.fill(syst2, lambda site: True, start=np.zeros(3));
    filmw = kwant.wraparound.wraparound(film)   
    filmw = filmw.finalized()

    M_trf = ft.partial(M_cubic, n=n)
    UM = UM_p(n)
    M = pg_op(filmw, M_trf, UM)

    pars = SnTe_params.copy()
    mSn = SnTe_params['m'][1]
    mTe = SnTe_params['m'][0]
    to_fd = filmw._wrapped_symmetry.to_fd

    pars['m'] = ft.partial(doped_m, doping=doping, mSn=mSn, mPb=mPb, mTe=mTe, n=n, to_fd=to_fd, salt=salt)
    # gap should be near the weighted average
    pars['mu'] = -(((1 - doping) * mSn + doping * mPb) + mTe) / 2
    pars['k_x'] = pars['k_y'] = pars['k_z'] = 0

    H = filmw.hamiltonian_submatrix(params=pars, sparse=True)
    ham_size = H.shape[0]
    norbs = filmw.sites[0].family.norbs

    x0, y0, z0 = position_operator(filmw)
    x = 1/np.sqrt(2) * (x0 - y0)
    y = z0

    # window half the size
    win_L11 = L11//2
    win_Lz = Lz//2
    A = win_L11 * win_Lz * np.sqrt(2)

    def shape1(site):
        tag = site.tag
        tag11 = np.dot(tag, n11) // np.dot(n11, n11)
        tagz = np.dot(tag, nz) // np.dot(nz, nz)
        tagn = (np.dot(tag, n) % (W * n.dot(n))) // np.dot(n, n)
        return (-win_L11/2 + L11/2 < tag11 <= win_L11/2 + L11/2 and
                -win_Lz/2 + Lz/2 < tagz <= win_Lz/2 + Lz/2 and
                tagn < W//2)
    window1 = make_window(filmw, shape1)
    
    def shape2(site):
        tag = site.tag
        tag11 = np.dot(tag, n11) // np.dot(n11, n11)
        tagz = np.dot(tag, nz) // np.dot(nz, nz)
        tagn = (np.dot(tag, n) % (W * n.dot(n))) // np.dot(n, n)
        return (-win_L11/2 + L11/2 < tag11 <= win_L11/2 + L11/2 and
                -win_Lz/2 + Lz/2 < tagz <= win_Lz/2 + Lz/2 and
                tagn >= W//2)
    window2 = make_window(filmw, shape2)

    return H, M, x, y, [window1, window2], pars, A
    

def job(doping, mPb):
    print(doping, mPb)
    
    H, M, x, y, windows, pars, A = make_operators(doping, mPb)
    
    spectrum = kwant.kpm.SpectralDensity(H, num_moments=num_moments, params=pars)

    es, dos = spectrum()
    ran = np.logical_and(-1 < es, es < 1)
    minimum = np.argmin(dos[ran])
    mine = es[ran][minimum]
    mindos = dos[ran][minimum]
    
    filling = spectrum.integrate(distribution_function=lambda x: x<mine) / spectrum.integrate()

    C_list = [mirror_chern(H, x, y, Mz=M, vectors=num_vectors,
                          e_F=mine, kpm_params=dict(num_moments=num_moments),
                          params=pars, bounds=None, window=window, return_std=False)
              for window in windows]

    C_list = np.array(C_list)
    Cs = np.sum(C_list, axis=0) / A
    C = np.mean(Cs)
    C_std = np.std(Cs)
    return C, C_std, filling, mine, mindos")
