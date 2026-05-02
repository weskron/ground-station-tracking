from utils.tle_convert import * 
from utils.frame_conversion import get_ground_station_location, ecf_to_tpc

# Simulation in km, kg, s

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
sigma_rho     = 5e-2 # km
sigma_rho_dot = 5e-3 # km/s
sigma_Az      = np.deg2rad(0.1)
sigma_El      = np.deg2rad(0.1)
R_k = np.diag([sigma_rho**2, sigma_rho_dot**2, sigma_Az**2, sigma_El**2]) # measurement noise

# Covariance Initialization
sigma_pos = 2 # km
sigma_vel = 0.1 # km/s
P0 = np.diag([sigma_pos**2, sigma_pos**2, sigma_pos**2, 
              sigma_vel**2, sigma_vel**2, sigma_vel**2])


# -- Ground Station Information ------
# LeoLabs global radar network - 7 operational sites (as of late 2024)

beta_max = np.deg2rad(80)  # max viewing angle

# --- New Zealand 
# Exact coords from radartutorial.eu: 45°02'19.9"S, 170°05'44.9"E
lat_nz = np.deg2rad(-45.0)
lon_nz = np.deg2rad(170.0)
r_ecf_nz = get_ground_station_location(R_EARTH, lat_nz, lon_nz)

# --- Alaska 
lat_ak = np.deg2rad(64.8)
lon_ak = np.deg2rad(-147.5)
r_ecf_ak = get_ground_station_location(R_EARTH, lat_ak, lon_ak)

# --- Texas 
lat_tx = np.deg2rad(31.9)
lon_tx = np.deg2rad(-102.2)
r_ecf_tx = get_ground_station_location(R_EARTH, lat_tx, lon_tx)

# --- 
lat_cr = np.deg2rad(10.0)
lon_cr = np.deg2rad(-84.2)
r_ecf_cr = get_ground_station_location(R_EARTH, lat_cr, lon_cr)

# --- 
lat_wa = np.deg2rad(-33.4)
lon_wa = np.deg2rad(116.1)
r_ecf_wa = get_ground_station_location(R_EARTH, lat_wa, lon_wa)

# --- Arizona
lat_az = np.deg2rad(36.9)
lon_az = np.deg2rad(-25.0)
r_ecf_az = get_ground_station_location(R_EARTH, lat_az, lon_az)

# --- Arizona 
lat_zn = np.deg2rad(32.5)
lon_zn = np.deg2rad(-109.5)
r_ecf_zn = get_ground_station_location(R_EARTH, lat_zn, lon_zn)


ground_stations = {
    "New Zealand": {
        "r_ecf": r_ecf_nz,
        "lat": lat_nz,
        "lon": lon_nz,
    },
    "Alaska": {
        "r_ecf": r_ecf_ak,
        "lat": lat_ak,
        "lon": lon_ak,
    },
    "Texas": {
        "r_ecf": r_ecf_tx,
        "lat": lat_tx,
        "lon": lon_tx,
    },
    "Costa Rica": {
        "r_ecf": r_ecf_cr,
        "lat": lat_cr,
        "lon": lon_cr,
    },
    "Western Australia": {
        "r_ecf": r_ecf_wa,
        "lat": lat_wa,
        "lon": lon_wa,
    },
    "Azores": {
        "r_ecf": r_ecf_az,
        "lat": lat_az,
        "lon": lon_az,
    },
    "Arizona": {
        "r_ecf": r_ecf_zn,
        "lat": lat_zn,
        "lon": lon_zn,
    },
}