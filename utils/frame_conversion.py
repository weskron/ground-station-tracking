import numpy as np

# Time
JD_epoch = 2461163.0 # Julian date at noon on 5/2/2026 this is the start of the sim
J2000 = 2451545.0
t0 = (2461163.0 - 2451545.0) / 36525.0 # simulation epoch in days

omega_earth = np.array([0, 0, 7.2921159e-5])  # rad/s, in ECF z-direction


def R3(theta):
    """Rotation about z-axis"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c,  s, 0],
        [-s,  c, 0],
        [ 0,  0, 1]
    ])

def eci_to_ecef(r_eci, gmst):
    """
    Simplified ECI -> ECEF using Earth rotation only
    gmst in radians
    """
    S = R3(gmst)
    return S @ r_eci

def ecef_to_eci(r_ecef, gmst):
    """
    Inverse transformation
    """
    S = R3(gmst)
    return S.T @ r_ecef

def vel_eci_to_ecef(v_eci, r_eci, gmst):
    R = R3(gmst)
    return R @ v_eci - np.cross(omega_earth, R @ r_eci)


def gmst_from_ut1(delta_T):
    """
    Compute GMST using your polynomial
    delta_T in seconds
    """
    gmst = (4.894961212823058 +
            delta_T * (6.300388098984893 +
            delta_T * (5.075209994113591e-15 -
            9.253097568194336e-24 * delta_T)))
    gmst = gmst % (2*np.pi)
    return gmst

def get_gmst(t_curr):
    JD = JD_epoch + t_curr / 86400.0
    t = (JD - J2000) / 36525.0
    delta_T = (JD - 2451545.0) * 86400.0
    gmst = gmst_from_ut1(delta_T)
    return gmst


def get_ground_station_location(R_EARTH, lat, lon):
    r_ecf = R_EARTH * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return r_ecf

def ecf_to_tpc(lat, lon):
    c_lat = np.cos(lat)
    s_lat = np.sin(lat)
    c_lon = np.cos(lon)
    s_lon = np.sin(lon)
    return np.array([
        [-s_lon, c_lon, 0],
        [-s_lat * c_lon, -s_lat * s_lon, c_lat],
        [c_lat * c_lon, c_lat * s_lon, s_lat]
    ])

def tpc_to_ecf(lat, lon):
    C_ET = ecf_to_tpc(lat, lon)
    return C_ET.T

