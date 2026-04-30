import numpy as np
import config
from utils.frame_conversion import *

def find_ground_station(gmst, x_true):
    r_sc_eci = x_true[0:3]
    # loop over all ground stations
    for name, r_gs_ecf in config.ground_stations.items():
        r_gs_eci = ecef_to_eci(r_gs_ecf, gmst) # ground station in intertial frame
        rho_eci = r_sc_eci - r_gs_eci
        rho_eci_norm = np.linalg.norm(rho_eci)
        rho_eci_hat = rho_eci / rho_eci_norm
        r_gs_eci_hat = r_gs_eci / config.R_EARTH
        beta = np.acos(np.dot(rho_eci_hat, r_gs_eci_hat))
        if beta <= config.beta_max:
            return name
        
    return "none"

def compute_measurement(gmst, gs_name, x_true):
    station = config.ground_stations[gs_name]
    r_gs_ecf = station["r_ecf"]
    lat = station["lat"]
    lon = station["lon"]

    r_sc_eci = x_true[0:3]
    v_sc_eci = x_true[0:3]
    r_sc_ecf = eci_to_ecef(r_sc_eci, gmst)
    
    # range
    rho_ecf = r_sc_ecf - r_gs_ecf
    rho = np.linalg.norm(rho_ecf)

    # range rate
    v_sc_ecf = vel_eci_to_ecef(v_sc_eci, r_sc_eci, gmst)
    rho_dot = np.dot(rho_ecf, v_sc_ecf) / rho

    # elevation
    rho_tpc = ecf_to_tpc(lat, lon) @ rho_ecf # range vector in the topocentric frame
    elevation = np.asin(rho_tpc[2] / rho)

    # azimuth 
    azimuth = np.arctan2(rho_tpc[0], rho_tpc[1]) % (2 * np.pi)


    return np.array([rho, rho_dot, elevation, azimuth])
