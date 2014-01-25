import numpy as np
import grumpy as gp

ee = 1.60218e-19
hbar = 1.05457e-34
kb = 1.38065e-23
mb = 9.274e-24

wr = 50e-9
tr = 50e-9
lr = 4 * 250e-9


def eso(diffc, lso):
    return hbar * diffc / (lso ** 2)


def i_typ(p, ec):
    return 4 * np.sqrt(3) * ec / (p ** 1.5 * 2 * np.pi * hbar / ee)


def b_m():
    return 0


def e_perp(r_width=None, r_thick=None, r_len=None):
    if r_width is None:
        r_width = wr
    if r_thick is None:
        r_thick = tr
    if r_len is None:
        r_len = lr

    return np.pi / 2 * (b_m() ** 2 * r_width * r_thick * r_len ** 2) / \
        ((2 * np.pi * hbar / ee) ** 2)


def Tp(p, ec):
    return 10.4 * ec / (p ** 2 * kb)


def ec(diffc, r_len=None):
    if r_len is None:
        r_len = lr
    return hbar * diffc / r_len ** 2


def ez(bmin, bmax):
    return mb / hbar * np.mean(np.array([bmin, bmax]))


def gd_integrand(x):
    prefactor = lambda e: (e * np.cosh(e) - np.sinh(e)) / (np.sinh(e) ** 3)
    bullshit = lambda x, y, e: np.real((1 + np.sqrt(x + 1j * y * e) +
                                       ((x + 1j * y * e) / 3)) *
                                       np.exp(-np.sqrt(x + 1j * y * e)))
    return lambda y, e: prefactor(e) * bullshit(x, y, e)


def gd(x, y):

    e_range = np.linspace(-30, 30, num=1e4)
    range_spacing = np.mean(np.diff(e_range))

    integrands = []

    x = gp.iterfy(x)
    y = gp.iterfy(y)

    if len(x) == 1:
        x = x * len(y)

    for x, values in zip(x, y):
        integrands.append(gd_integrand(x)(values, e_range))

    integrand_values = np.vstack(tuple(integrands))

    return np.sum(integrand_values, axis=1) * range_spacing


def pcmag(p, bmin, bmax, diffc, lso, temp, r_width=None, r_thick=None,
          r_len=None):

    Ec = ec(diffc, r_len)
    Eperp = e_perp(r_width, r_thick, r_len)
    Eso = eso(diffc, lso)
    Ez = ez(bmin, bmax)

    tscale = 20.8 * np.array(temp) / Tp(p, Ec)

    tfactor = gd(p ** 2 * Eperp, tscale)
    sofactor = gd(p ** 2 * (Eperp + 4/3 * Eso / Ec), tscale)

    magfactor1 = gd(p ** 2 * (Eperp + 4/3 * Eso / Ec -
                    2j * Ez / Ec), tscale)
    magfactor2 = gd(p ** 2 * (Eperp + 4/3 * Eso / Ec +
                    2j * Ez / Ec), tscale)

    pc = i_typ(p, Ec) * (1 / np.sqrt(2)) * np.sqrt(tfactor + sofactor +
                                                   magfactor1 + magfactor2)
    pc[np.isnan(pc)] = 0
    return pc
