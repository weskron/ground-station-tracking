from utils.tle_convert import * 
from utils.frame_conversion import get_ground_station_location, ecf_to_tpc

# Simulaiton in km, kg, s, 

# -- Simulation Parameters
MU_EARTH = 3.986e5 # km^3/s^2
R_EARTH = 6371.8
J2 = 1.0826e-3


# -- Target TLE -------------------
tle1 = "1 37766U 11039B   26118.76594549  .00000637  00000-0  11237-3 0  9993"
tle2 = "2 37766  98.1103  16.7383 0043964 107.3269 253.2753 14.70311174789533"

r, v = tle_to_cartesian(tle1, tle2)
x_0 = np.concatenate([r, v])

# print("Position (km):", r)
# print("Velocity (km/s):", v)

# -- Statistical initialization -----
# Process Noise
sigma_accel = 1.08263e-3 # km/s^2, Size of J2 acceleration
Q_s = np.diag([sigma_accel**2, sigma_accel**2, sigma_accel**2]) # process noise 

# Measurement Noise
sigma_rho     = 50 # km
sigma_rho_dot = 0.5 # km/s
sigma_Az      = np.deg2rad(0.1)
sigma_El      = np.deg2rad(0.1)
R_k = np.diag([sigma_rho**2, sigma_rho_dot**2, sigma_Az**2, sigma_El**2]) # measurement noise

# Covariance Initialization
sigma_pos = 2 # km
sigma_vel = 0.1 # km/s
P0 = np.diag([sigma_pos**2, sigma_pos**2, sigma_pos**2, 
              sigma_vel**2, sigma_vel**2, sigma_vel**2])


# -- Ground Station Information ------
beta_max = np.deg2rad(50) # max viewing angle
# New Zealand
lat_nz = np.deg2rad(-45.0)
lon_nz = np.deg2rad(170.0)
r_ecf_nz = get_ground_station_location(R_EARTH, lat_nz, lon_nz)
r_tpc_nz = ecf_to_tpc(lat_nz, lon_nz) @ r_ecf_nz

# Alaska

ground_stations = {
    "New Zealand": {
        "r_ecf": r_ecf_nz,
        "lat": lat_nz,
        "lon": lon_nz,
    },
}

