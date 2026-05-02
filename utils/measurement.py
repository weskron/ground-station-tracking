import numpy as np
import config
from utils.frame_conversion import *

def find_ground_station(gmst, x_true):
    r_sc_eci = x_true[0:3]

    for name, station in config.ground_stations.items():
        r_gs_ecf = station["r_ecf"]       
        lat      = station["lat"]
        lon      = station["lon"]

        r_gs_eci     = ecef_to_eci(r_gs_ecf, gmst)
        rho_eci      = r_sc_eci - r_gs_eci
        rho_eci_norm = np.linalg.norm(rho_eci)
        rho_eci_hat  = rho_eci / rho_eci_norm
        r_gs_eci_hat = r_gs_eci / np.linalg.norm(r_gs_eci)

        beta = np.arccos(np.clip(np.dot(rho_eci_hat, r_gs_eci_hat), -1, 1))  # clip guards arccos domain
        if beta <= config.beta_max:
            return name

    return "none"

def compute_measurement(gmst, gs_name, x_true):
    station = config.ground_stations[gs_name]
    r_gs_ecf = station["r_ecf"]
    lat = station["lat"]
    lon = station["lon"]
    r_sc_eci = x_true[0:3]
    v_sc_eci = x_true[3:6]

    # -- Positions in ECEF ------------
    r_sc_ecf = eci_to_ecef(r_sc_eci, gmst)
    v_sc_ecf = vel_eci_to_ecef(v_sc_eci, r_sc_eci, gmst)

    # Relative position and velocity
    rho_ecf = r_sc_ecf - r_gs_ecf
    v_rel = v_sc_ecf
    
    # -- Range -------------------------
    rho = np.linalg.norm(rho_ecf)
   
    # -- Range Rate --------------------
    rho_dot = np.dot(rho_ecf, v_rel) / rho

    # -- NEU Components ----------------
    R_neu = ecf_to_tpc(lat, lon)
    rho_tpc = R_neu @ rho_ecf # range vector in the topocentric frame
    n_comp, e_comp, u_comp = rho_tpc
    
    elevation = float(np.arcsin(np.clip(u_comp / rho, -1.0, 1.0)))

    # azimuth 
    azimuth = float(np.arctan2(e_comp, n_comp) % (2.0 * np.pi))

    return np.array([rho, rho_dot, elevation, azimuth])

