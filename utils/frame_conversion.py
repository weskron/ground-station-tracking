import numpy as np
from datetime import datetime, timezone

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



def julian_date(dt: datetime) -> float:
    """Return the Julian Date for a UTC datetime."""
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return 2_451_545.0 + (dt - j2000).total_seconds() / 86_400.0


def gmst_from_datetime(dt: datetime) -> float:
    """
    Compute Greenwich Mean Sidereal Time (GMST) in radians.
    Uses the IAU 1982 formula accurate to ~0.1 arcsec over several decades.
    """
    jd = julian_date(dt)
    T  = (jd - 2_451_545.0) / 36_525.0
    gmst_deg = (100.4606184
                + 36_000.77004 * T
                + 0.000387933 * T**2
                - T**3 / 38_710_000.0)
    ut1_seconds = (dt.hour * 3600 + dt.minute * 60
                   + dt.second + dt.microsecond * 1e-6)
    gmst_deg += 360.98564724 * ut1_seconds / 86_400.0
    return np.deg2rad(gmst_deg % 360.0)

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def get_ground_station_location(R_EARTH, lat, lon):
    r_ecf = R_EARTH * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return r_ecf

def ecf_to_tpc(lat_rad, lon_rad):
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

    # Unit vectors of the NEU frame expressed in ECEF
    N = np.array([-sin_lat * cos_lon,
                  -sin_lat * sin_lon,
                   cos_lat])
    E = np.array([-sin_lon,
                   cos_lon,
                   0.0])
    U = np.array([ cos_lat * cos_lon,
                   cos_lat * sin_lon,
                   sin_lat])

    return np.vstack([N, E, U])

def tpc_to_ecf(lat, lon):
    C_ET = ecf_to_tpc(lat, lon)
    return C_ET.T

