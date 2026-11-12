"""NLO F2 of the pi- for many x_pi at once, from a PDF tabulated on a fixed x grid.
Runs on the GPU with cupy if it is installed, otherwise with numpy.
"""

import numpy as np

# Device backend
try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False


def backend_name():
    return "GPU (CuPy)" if GPU else "CPU (NumPy, vectorised)"


def to_gpu(a):
    return xp.asarray(a)


def to_cpu(a):
    return xp.asnumpy(a) if GPU else np.asarray(a)


# Constants -- must match the scalar code in h1_five_settings.py
_CF = 4.0 / 3.0
_TR = 0.5

_E2_u = (2 / 3) ** 2
_E2_d = (1 / 3) ** 2
_E2_s = (1 / 3) ** 2
_E2_c = (2 / 3) ** 2
_E2_b = (1 / 3) ** 2
_CHARGE_SQ_SUM = _E2_u + _E2_d + _E2_s + _E2_c + _E2_b

# pi^- flavour assignment
_FLAVOR_MAP_PIMINUS = [
    (1, -1, _E2_u),
    (2, -2, _E2_d),
    (3, -3, _E2_s),
    (4, -4, _E2_c),
    (5, -5, _E2_b),
]

_Cq2_DELTA = _CF * (np.pi ** 2 / 3.0 - 9.0 / 2.0)

PIDS_NEEDED = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 21]

# Tabulation grid. Log-spaced, and interpolation is done in log x, which is
# where PDFs are smooth.
X_FINE = np.logspace(-7, np.log10(1.0 - 1e-8), 6000)
_LOG_X_FINE = np.log(X_FINE)

# Fixed-order Gauss-Legendre nodes. Generous, since raising it is nearly
# free once the work is batched.
N_QUAD = 256
_GL_N, _GL_W = np.polynomial.legendre.leggauss(N_QUAD)
GL_NODES = to_gpu(_GL_N)
GL_WEIGHTS = to_gpu(_GL_W)

_ZMAX = 1.0 - 1e-6      # upper limit of the z integral


class PionPdfGrid:
    """Tabulates one PDF member at one Q^2 (CPU, via LHAPDF) and interpolates
    xf(pid, x) for arbitrary batched x arrays on the device.
    """

    def __init__(self, pdf, q2val):
        self.q2 = float(q2val)
        grids = {}
        for pid in PIDS_NEEDED:
            vals = np.empty_like(X_FINE)
            for k, xx in enumerate(X_FINE):
                try:
                    vals[k] = pdf.xfxQ2(int(pid), float(xx), self.q2)
                except Exception:
                    vals[k] = 0.0
            grids[pid] = vals
        self.log_x = to_gpu(_LOG_X_FINE)
        self.grids = {pid: to_gpu(g) for pid, g in grids.items()}

    def xfq(self, pid, xval):
        """xf(pid, x), 0 outside (0, 1). xval may be any shape."""
        shape = xval.shape
        flat = xval.reshape(-1)
        valid = (flat > 0.0) & (flat < 1.0)
        safe = xp.where(valid, flat, 1e-30)
        out = xp.interp(xp.log(safe), self.log_x, self.grids[pid],
                        left=0.0, right=0.0)
        out = xp.where(valid, out, 0.0)
        return out.reshape(shape)


def _Cq2_singular(z):
    """CF (1+z^2) (ln((1-z)/z) - 3/4), vectorised, 0 outside (0,1)."""
    inside = (z > 0.0) & (z < 1.0)
    zs = xp.where(inside, z, 0.5)
    val = _CF * (1.0 + zs * zs) * (xp.log((1.0 - zs) / zs) - 0.75)
    return xp.where(inside, val, 0.0)


def _Cg2_kernel(z):
    """2 TR [ (z^2+(1-z)^2) ln((1-z)/z) - 1 + 8 z (1-z) ]."""
    inside = (z > 0.0) & (z < 1.0)
    zs = xp.where(inside, z, 0.5)
    split = (zs * zs + (1.0 - zs) ** 2) * xp.log((1.0 - zs) / zs)
    poly = -1.0 + 8.0 * zs * (1.0 - zs)
    return xp.where(inside, 2.0 * _TR * (split + poly), 0.0)


def _gl_nodes_on(a, b):
    """Gauss-Legendre nodes/jacobian for a batch of intervals [a_i, b_i].
    a, b are 1-D device arrays of length n. Returns (z, w, jac) with z of
    shape (n, N_QUAD).
    """
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    z = mid[:, None] + half[:, None] * GL_NODES[None, :]
    return z, GL_WEIGHTS[None, :], half[:, None]


def f2pi_minus_nlo_vec(pdf, grid, x_arr_cpu, q2val):
    """NLO F2^{pi-} for every x in x_arr_cpu (1-D numpy) at one Q^2, using the
    tabulated `grid`. Returns a numpy array of the same length.
    """
    x_cpu = np.asarray(x_arr_cpu, dtype=np.float64)
    x = to_gpu(x_cpu)
    n = x.shape[0]

    pref = float(pdf.alphasQ(np.sqrt(float(q2val)))) / (2.0 * np.pi)

    in_range = (x > 0.0) & (x < 1.0)
    x_safe = xp.where(in_range, x, 0.5)

    # z integrals over [x, 1-1e-6]
    zmax = xp.full(n, _ZMAX, dtype=xp.float64)
    z, w, jac = _gl_nodes_on(x_safe, zmax)
    one_minus_z = 1.0 - z
    Cs = _Cq2_singular(z)                       # (n, N)
    xo = x_safe[:, None] / z                    # PDF argument x/z
    valid_xo = xo < 1.0

    # boundary term  bdy(x) = int_0^x C(z)/(1-z) dz
    zb, wb, jacb = _gl_nodes_on(xp.zeros(n, dtype=xp.float64), x_safe)
    bdy = xp.sum(jacb * wb * (_Cq2_singular(zb) / (1.0 - zb)), axis=1)

    f2 = xp.zeros(n, dtype=xp.float64)

    for pid_q, pid_qb, e2 in _FLAVOR_MAP_PIMINUS:
        fx = grid.xfq(pid_q, x_safe) + grid.xfq(pid_qb, x_safe)
        fz = grid.xfq(pid_q, xo) + grid.xfq(pid_qb, xo)
        fz = xp.where(valid_xo, fz, 0.0)

        # leading order
        f2 = f2 + e2 * fx

        # "+"-distribution piece
        integ_plus = Cs * (fz - fx[:, None]) / one_minus_z
        I_plus = xp.sum(jac * w * integ_plus, axis=1) - fx * bdy

        # regular piece  CF (3/2 + 2z) f(x/z)
        integ_reg = _CF * (1.5 + 2.0 * z) * fz
        reg = xp.sum(jac * w * integ_reg, axis=1)

        # delta piece
        delta = _Cq2_DELTA * fx

        f2 = f2 + e2 * pref * (I_plus + reg + delta)

    # gluon
    g = grid.xfq(21, xo)
    g = xp.where(valid_xo, g, 0.0)
    gluon = xp.sum(jac * w * (_Cg2_kernel(z) * g), axis=1)
    f2 = f2 + pref * _CHARGE_SQ_SUM * gluon

    f2 = xp.where(in_range, f2, 0.0)
    return to_cpu(f2)


def rebin_pion_to_xl_edges_fast(pdf_obj, x_means, q2_means, xl_edges,
                                n_xl, flux_fn, n_sub=12, label=""):
    """Batched version of rebin_pion_to_xl_edges in h1_five_settings.py."""
    import time
    t0 = time.time()

    # x_L sub-sample points, shared by every cell
    sub_pts = []
    for i in range(n_xl):
        lo, hi = xl_edges[i], xl_edges[i + 1]
        sub_pts.append(lo + (np.arange(n_sub) + 0.5) * (hi - lo) / n_sub)
    sub_pts = np.concatenate(sub_pts)             # (n_xl * n_sub,)

    # LN flux factor -- depends only on x_L, so compute it once.
    flux_sub = np.array([2.0 * float(flux_fn(float(xl))) for xl in sub_pts])
    flux_grid = flux_sub.reshape(n_xl, n_sub)

    out = {}
    for iq, q2 in enumerate(q2_means):
        grid = PionPdfGrid(pdf_obj, float(q2))

        # every x_pi this Q^2 needs, flattened
        x_pi_all = []
        for xbj in x_means:
            x_pi_all.append(xbj / (1.0 - sub_pts))
        x_pi_all = np.concatenate(x_pi_all)       # (n_x * n_xl * n_sub,)

        vals = f2pi_minus_nlo_vec(pdf_obj, grid, x_pi_all, float(q2))
        vals = vals.reshape(len(x_means), n_xl, n_sub)

        # apply the LN flux, then average over the sub-points in each x_L bin
        vals = vals * flux_grid[None, :, :]

        for ix in range(len(x_means)):
            out[(iq, ix)] = vals[ix].mean(axis=1)

        print(f"    [{label}] Q2={q2:g} done "
              f"({len(x_pi_all):,} x_pi points)  "
              f"elapsed {time.time() - t0:5.1f}s", flush=True)

    return out
