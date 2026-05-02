import sympy as sp
import numpy as np
import config

# --- Symbolic state ---
rx, ry, rz, vx, vy, vz = sp.symbols('rx ry rz vx vy vz', real=True)
r_eci = sp.Matrix([rx, ry, rz])
v_eci = sp.Matrix([vx, vy, vz])

# --- Symbolic constants (treated as known at filter time) ---
gmst   = sp.Symbol('gmst', real=True)
lat    = sp.Symbol('lat', real=True)
lon    = sp.Symbol('lon', real=True)
gsx, gsy, gsz = sp.symbols('gsx gsy gsz', real=True)  # r_gs_ecf
omega_e = sp.Symbol('omega_e', positive=True)          # Earth rotation rate

# --- ECI -> ECEF rotation matrix ---
C = sp.Matrix([
    [ sp.cos(gmst), sp.sin(gmst), 0],
    [-sp.sin(gmst), sp.cos(gmst), 0],
    [0,             0,            1]
])

# --- ECF -> Topocentric (NEU ) ---
M = sp.Matrix([
    [-sp.sin(lat)*sp.cos(lon), -sp.sin(lat)*sp.sin(lon),  sp.cos(lat)],  # N
    [-sp.sin(lon),              sp.cos(lon),              0],            # E
    [ sp.cos(lat)*sp.cos(lon),  sp.cos(lat)*sp.sin(lon),  sp.sin(lat)]   # U
])

# --- Earth rotation vector ---
omega = sp.Matrix([0, 0, omega_e])

# --- Derived quantities ---
r_gs_ecf = sp.Matrix([gsx, gsy, gsz])
r_ecf    = C @ r_eci
v_ecf    = C @ v_eci - omega.cross(r_ecf)

v_rel    = v_ecf


rho_ecf  = r_ecf - r_gs_ecf
rho      = sp.sqrt(rho_ecf.dot(rho_ecf))

rho_dot  = rho_ecf.dot(v_rel) / rho

rho_tpc  = M @ rho_ecf

n_comp = rho_tpc[0]
e_comp = rho_tpc[1]
u_comp = rho_tpc[2]

azimuth = sp.atan2(e_comp, n_comp)

elevation = sp.asin(u_comp / rho)

# --- Measurement vector ---
z = sp.Matrix([rho, rho_dot, elevation, azimuth])

# --- State vector ---
x = sp.Matrix([rx, ry, rz, vx, vy, vz])

# --- Compute Jacobian (this is the slow step) ---
# print("Computing Jacobian symbolically... (may take 30-60s)")
H_sym = z.jacobian(x)

# Substitute your actual numeric constants for omega_e, lat, lon, gmst, gs position
# at runtime — lambdify turns the symbolic expression into a fast numpy function

H_func = sp.lambdify(
    [rx, ry, rz, vx, vy, vz, gmst, lat, lon, gsx, gsy, gsz, omega_e],
    H_sym,
    modules='numpy'
)

# Usage inside your EKF:
def compute_H(gmst_val, gs_name, x_est):
    station = config.ground_stations[gs_name]
    r_gs    = station["r_ecf"]
    return np.array(H_func(
        *x_est,
        gmst_val,
        station["lat"], station["lon"],
        r_gs[0], r_gs[1], r_gs[2],
        7.2921150e-5
    ), dtype=float)