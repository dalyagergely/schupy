import numpy as np
import matplotlib.pyplot as plt
from random import random

eps = 8.8541878e-12
mu = 4.0e-7 * np.pi
R = 6371000.0


def plot_map(
    s_lat, s_lon, s_int, m_lat, m_lon, show=True, save=False, filename="schupy_map.png"
):
    import cartopy.crs as ccrs
    
    assert len(s_lat) == len(s_lon), 's_lat should have the same number of ' \
                                     'elements as s_lon'

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.text(
        -0.12,
        0.55,
        "Latitude",
        va="bottom",
        ha="center",
        rotation="vertical",
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    
    ax.text(
        0.5,
        -0.2,
        "Longitude",
        va="bottom",
        ha="center",
        rotation="horizontal",
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    
    ax.coastlines()
    ax.stock_img()

    ax.scatter(m_lon, m_lat, c="cyan", marker="s", s=100, edgecolors="k", alpha=1)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="k", alpha=1
    )

    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False

    for lon, lat in zip(s_lon, s_lat):
        ax.scatter(
            lon,
            lat,
            c='orange',
            s=80,
            alpha=1
        )

    plt.tight_layout()
    
    if show:
        plt.show()
        
    if save:
        plt.savefig(filename, dpi=300)
        
    plt.close()


def height_calculation(
    freq,
    fkn=10.0,
    hkn=55000,
    kszib=8300,
    kszia=2900,
    fm=8.0,
    hm=96500,
    kszim=4000,
    bm=6500,
):
    """
    Mushtak and Williams (2002)
    """
    
    freq = np.asarray(freq)
    assert freq.ndim == 1, 'Expected 1D array freq.'
    nfreq = len(freq)
    
    Re_He = (
            hkn
            + kszia * np.log(freq / fkn)
            + 0.5 * (kszia - kszib) * np.log(1 + (fkn / freq) ** 2)
        )

    Im_He = -0.5 * np.pi * kszia + (kszia - kszib) * np.arctan(fkn / freq)
    Re_Hm = hm - (kszim + bm * (1 / freq - 1 / fm)) * np.log(freq / fm)
    Im_Hm = 0.5 * np.pi * (kszim + bm * (1 / freq - 1 / fm))
    
    He = Re_He + 1j * Im_He
    Hm = Re_Hm + 1j * Im_Hm

    return He, Hm


def height_calculation_kul(freq):
    """
    Kulak and Mlynarczyk (2013)
    """
    
    freq = np.asarray(freq)
    assert freq.ndim == 1, 'Expected 1D array freq.'
    nfreq = len(freq)
    
    Re_He_n = (
        67.5
        + 2 * np.log(freq / 7.7)
        - 2.54 * (7.7 / freq) ** 0.813
        - 2.72 * (7.7 / freq) ** 1.626
    )
    
    Im_He_n = (
        -3.14 - 8.70 * (7.7 / freq) ** 0.813 + 1.92 * (7.7 / freq) ** 1.626
    )

    Re_Hm_n = 114.7 - 8.4 * np.log(freq / 7.7)
    Im_Hm_n = 13.2 - 2.0 * np.log(freq / 7.7)

    Re_He_d = (
        51.1
        + 1.9 * np.log(freq / 1.7)
        - 2.45 * (1.7 / freq) ** 0.822
        - 2.84 * (1.7 / freq) ** 1.645
    )
    
    Im_He_d = (
        -2.98 - 8.80 * (1.7 / freq) ** 0.822 + 1.86 * (1.7 / freq) ** 1.645
    )
    
    Re_Hm_d = 101.5 - 3.1 * np.log(freq / 7.7)
    Im_Hm_d = 7.0 - 0.9 * np.log(freq / 7.7)
    
    HE_n = 1000 * (Re_He_n + 1j * Im_He_n)
    HM_n = 1000 * (Re_Hm_n + 1j * Im_Hm_n)
    HE_d = 1000 * (Re_He_d + 1j * Im_He_d)
    HM_d = 1000 * (Re_Hm_d + 1j * Im_Hm_d)

    return 0.5 * (HE_n + HE_d), 0.5 * (HM_n + HM_d)


def ZYR2_cal(freq, R):
    He, Hm = (
        height_calculation(freq)
        if height == "mushtak"
        else height_calculation_kul(freq)
    )
    
    C = eps / He
    L = mu * Hm
    Z = 1j * 2 * np.pi * freq * L
    Y = -1j * 2 * np.pi * freq * C

    return Y * Z * R * R


def greens(freq, R, xs, ps, xm, pm, n):
    ZYR2 = ZYR2_cal(freq, R)
    
    cg = xm * xs + np.sqrt(1 - xm * xm) * np.sqrt(1 - xs * xs) * np.cos(pm - ps)
    p0 = 1
    p1 = cg
    gr = -p0 / ZYR2
    gr = gr + 3.0 * p1 / (2.0 - ZYR2)
    for n in range(2, n):
        pn = ((2 * n - 1.0) * p1 * cg - (n - 1) * p0) / n
        grp = (2 * n + 1.0) * pn / (n * (n + 1) - ZYR2)
        gr = gr + grp
        p0 = p1
        p1 = pn

    return gr / (4 * np.pi)


def greens_d(freq, R, xs, ps, xm, pm, n, t):
    ZYR2 = ZYR2_cal(freq, R)
    cg = xm * xs + np.sqrt(1 - xm * xm) * np.sqrt(1 - xs * xs) \
         * np.cos(pm - ps)
         
    p0 = 1.0
    p0d = 0.0
    p1 = cg
    p1d = 1.0
    
    gr = -p0d / ZYR2
    gr = gr + 3.0 * p1d / (2.0 - ZYR2)
    
    for n in range(2, n):
        pn = ((2 * n - 1.0) * p1 * cg - (n - 1) * p0) / n
        pnd = (2 * n - 1.0) * p1 + p0d
        grp = (2 * n + 1.0) * pnd / (n * (n + 1.0) - ZYR2)
        gr = gr + grp
        p0 = p1
        p1 = pn
        p0d = p1d
        p1d = pnd

    if t == 1:
        gr = gr * (
            -np.sqrt(1.0 - xm * xm) * xs
            + xm * np.sqrt(1.0 - xs * xs) * np.cos(pm - ps)
        )
    if t == 2:
        gr = (
            -gr
            * np.sqrt(1.0 - xm * xm)
            * np.sqrt(1.0 - xs * xs)
            * np.sin(pm - ps)
        )

    return gr / (4 * np.pi)


def extended(s_lat, s_lon, s_int, radius, R, num=100):

    def rotation_matrix(axis, theta):
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        
        return np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )

    def StoC(lat, lon):
        theta = 90 - lat
        if lon < 0:
            phi = 360 + lon
        else:
            phi = lon

        return [
            np.sin(np.radians(theta)) * np.cos(np.radians(phi)),
            np.sin(np.radians(theta)) * np.sin(np.radians(phi)),
            np.cos(np.radians(theta)),
        ]

    def CtoS(r):
        x = r[0]
        y = r[1]
        z = r[2]
        theta = np.degrees(np.arccos(z / np.linalg.norm(r)))
        phi = np.degrees(np.arctan2(y, x))

        lat = 90 - theta
        if phi > 180:
            lon = phi - 360
        else:
            lon = phi

        return lat, lon

    def Delta(r1, r2):
        return np.degrees(np.arccos(np.dot(r1, r2)))

    for k in range(len(s_lat)):
        phi = []
        theta = []

        lim = 1 - np.cos(radius * 1000000 / R)

        for i in range(num):
            phi.append(random() * 2 * np.pi)
            theta.append(np.arccos(1 - random() * lim))

        lat1 = []
        lon1 = []

        for i in range(num):
            lat1.append(90 - np.degrees(theta[i]))
            if phi[i] > np.pi:
                lon1.append(np.degrees(phi[i]) - 360)
            else:
                lon1.append(np.degrees(phi[i]))

        rotation_ax = np.cross(StoC(90, 0), StoC(s_lat[k], s_lon[k]))
        gamma = Delta(StoC(90, 0), StoC(s_lat[k], s_lon[k]))

        lat_new = np.zeros(num)
        lon_new = np.zeros(num)

        for i in range(num):
            [lat_new[i], lon_new[i]] = CtoS(
                rotation_matrix(rotation_ax, np.radians(gamma)).dot(
                    StoC(lat1[i], lon1[i])
                )
            )

            s_lon_ext.append(lon_new[i])
            s_lat_ext.append(lat_new[i])
            s_int_ext.append(s_int[k] / num)


def forward_tdte(
    s_lat,
    s_lon,
    s_int,
    m_lat,
    m_lon,
#    f_min=5.0,
#    f_max=30.0,
#    f_step=0.1,
    freq,
    radius=0,
    n=500,
    h="mushtak",
    ret="all",
    mapshow=False,
    mapsave=False,
    mapfilename="schupy_map.png",
    plotshow=False
):

    try:
        len(s_lat)
    except TypeError:
        s_lat = [s_lat]
        s_lon = [s_lon]
        s_int = [s_int]
        
    try:
        len(freq)
    except TypeError:
        freq = [freq]
        
    for i in range(len(s_int)):
        s_int[i] = s_int[i] * 1e6

    assert len(s_lat) == len(s_lon) or len(s_lat) == len(s_int), \
            "s_lat, s_lon and s_int should have the same number of elements."
    
    if h not in ["mushtak", "kulak"]:
        raise ValueError("Height calculation should be set either to mushtak or kulak.")
        
    if ret not in [
        "all",
        "ER",
        "Er",
        "er",
        "BP",
        "Bp",
        "bp",
        "bphi",
        "Bphi",
        "BT",
        "Bt",
        "bt",
        "btheta",
        "Btheta",
    ]:
        raise ValueError(
            "Returned value could either be Er (set ER, Er, or er), "
            "Bphi (set BP, Bp, bp, Bphi or bphi), "
            "Bt (set BT, Bt, bt, Btheta or btheta) or all of the above (set all)."
        )
        
    if radius < 0:
        raise ValueError(
            "Radius should be 0 for point sources and a positive number for extended sources."
        )

    if radius > 0:
        global s_lon_ext
        global s_lat_ext
        global s_int_ext

        s_lon_ext = []
        s_lat_ext = []
        s_int_ext = []

        extended(s_lat, s_lon, s_int, radius=radius, R=R)

        s_lon = s_lon_ext
        s_lat = s_lat_ext
        s_int = s_int_ext

    if mapshow == True or mapsave == True:
        plot_map(s_lat, s_lon, s_int, m_lat, m_lon, show=mapshow, save=mapsave, filename=mapfilename)
        

    #freq = np.arange(f_min, f_max, f_step)
    omega = 2 * np.pi * freq

    global height
    height = "mushtak" if h == "mushtak" else "kulak"
    he, hm = (
        height_calculation(freq)
        if height == "mushtak"
        else height_calculation_kul(freq)
    )

    Ez = np.zeros(len(freq))
    Bph = np.zeros(len(freq))
    Bt = np.zeros(len(freq))

    for s in range(len(s_lat)):
        ez = greens(
            freq,
            R,
            np.cos(np.radians(90 - s_lat[s])),
            np.radians(s_lon[s]),
            np.cos(np.radians(90.0 - float(m_lat))),
            np.radians(float(m_lon)),
            n,
        )
        ez = np.abs(1000 * ez * (1j * omega * mu * hm) / he ** 2) ** 2
        Ez = Ez + ez * s_int[s]

        bph = greens_d(
            freq,
            R,
            np.cos(np.radians(90 - s_lat[s])),
            np.radians(s_lon[s]),
            np.cos(np.radians(90.0 - float(m_lat))),
            np.radians(float(m_lon)),
            n,
            1,
        )
        bph = np.abs(1.0e12 * bph * (mu) / (he * R)) ** 2
        Bph = Bph + bph * s_int[s]

        bt = greens_d(
            freq,
            R,
            np.cos(np.radians(90 - s_lat[s])),
            np.radians(s_lon[s]),
            np.cos(np.radians(90.0 - float(m_lat))),
            np.radians(float(m_lon)),
            n,
            2,
        )
        bt = (
            np.abs(
                1e12 * bt * mu / (he * R * np.sin(np.radians(90 - float(m_lat))))
            )
            ** 2
        )
        Bt = Bt + bt * s_int[s]

    if plotshow == True:
        f_min = min(freq)
        f_max = max(freq)
        plt.rcParams["figure.figsize"] = [12, 7]
        plt.subplot(3, 1, 1)
        plt.title("Obsevation site: " + r"$(%d^o,%d^o)$" % (m_lat, m_lon), fontsize=18)
        plt.plot(freq, Ez)
        plt.ylabel(r"$E_r\ [mV^2/m^2/Hz]$", fontsize=16)
        plt.xticks(
            np.arange(f_min - 1, f_max + 3, 2), len(np.arange(f_min - 1, f_max + 3, 2)) * []
        )
        plt.xlim([f_min - 1, f_max + 1])
        plt.tick_params(axis="both", which="major", labelsize=14)
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(freq, Bt)
        plt.ylabel(r"$B_{\vartheta}\ [pT^2/Hz]$", fontsize=16)
        plt.xticks(
            np.arange(f_min - 1, f_max + 3, 2), len(np.arange(f_min - 1, f_max + 3, 2)) * []
        )
        plt.xlim([f_min - 1, f_max + 1])
        plt.tick_params(axis="both", which="major", labelsize=14)
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(freq, Bph)
        plt.ylabel(r"$B_{\varphi}\ [pT^2/Hz]$", fontsize=16)
        plt.xticks(np.arange(f_min - 1, f_max + 3, 2))
        plt.xlim([f_min - 1, f_max + 1])
        plt.xlabel("Frequency [Hz]", fontsize=16)
        plt.tick_params(axis="both", which="major", labelsize=14)
        plt.grid()

        plt.tight_layout()
        plt.show()

    if ret == "all":
        return Ez, Bt, Bph
    elif ret in ["ER", "Er", "er"]:
        return Ez
    elif ret in ["BP", "Bp", "bp", "Bphi", "bphi"]:
        return Bph
    elif ret in ["BT", "Bt", "bt", "Btheta", "btheta"]:
        return Bt
