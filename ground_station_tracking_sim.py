import numpy as np
from config import MU_EARTH, R_EARTH, x_0, P0, Q_s, R_k
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from utils.dynamics import spacecraft_dyn, spacecraft_dyn_stm
from utils.frame_conversion import *
from utils.measurement import *
from utils.measurement_jacobian import compute_H
from datetime import datetime, timedelta, timezone

np.random.seed(0)

def main():
    t0 = datetime(2026, 4, 30, 0, 0, 0, tzinfo=timezone.utc)
    tf = 1 * 24 * 60 * 60 # time -> seconds
    dt = 15 # seconds
    N = int(tf / dt)
    t_sim = np.linspace(0, tf, N)

    # -- Kalman Filter Initializations ---------
    nx = 6
    nz = 4

    # gmst preallocations
    gmst = np.zeros(N)

    # Measurement preallocations
    z_hat_k = np.zeros((nz, N)) # measurement based on estimated state
    z_k = np.zeros((nz, N)) # nonlinear measurement using true target state
    viewing_station = []
    # State preallocations
    x_true = np.zeros((nx, N+1))
    x_true[:, 0] = x_0

    noise0 = np.random.multivariate_normal(np.zeros(6), P0).flatten()
    xmkm = np.zeros((nx, N+1)) 
    xmkm[:, 0] = x_0 + noise0
    xmkp = np.zeros((nx, N+1))
    xmkp[:, 0] = x_0 + noise0

    # Covariance preallocations
    Pmkm = []
    Pmkm.append(P0)
    Pmkp = []
    Pmkp.append(P0)

    # State Transition Matrix Preallocation

    # EKF preallocations
    W_k = [] # innovations covariance
    C_k = [] # cross covariance
    K_k = [] # Kalman Gain
    H_k = [] # measurement mapping

    # Noise preallocations
    nu = np.random.multivariate_normal(np.zeros(4), R_k, N) # preallocate the noise for the measurements

    # Noise Mapping Matrix
    G = np.vstack([np.zeros((3, 3)),
                  np.eye(3)])

    # -- Simulation --------------

    for k in range(N):
        tk   = k * dt
        tkp1 = (k+1) * dt

        # Update gmst and eci-ecf transition
        tgmst = t0 + timedelta(seconds=(k+1) * dt)
        gmst[k] = gmst_from_datetime(tgmst)
        # print(gmst[k])

        # propagate true target state w/ noise
        true_traj = solve_ivp(
            spacecraft_dyn,
            [tk, tkp1],
            x_true[:, k],
            args=(np.zeros(3), ),
            atol=1e-12, rtol=1e-12,
        )
        x_true[:, k+1] = true_traj.y[:, -1]

        # Complete the predict step of the EKF
        predict_ekf = solve_ivp(
            spacecraft_dyn_stm,
            [tk, tkp1],
            np.concatenate([xmkp[:, k], Pmkp[k].flatten()]),
            args=(G, Q_s),
            atol=1e-12, rtol=1e-12,
        )
        xmkm[:, k+1] = predict_ekf.y[0:6, -1]
        Pmkm.append(predict_ekf.y[6:(6+6**2), -1].reshape(6,6))

        # determine if the target is within the viewing cone of the ground stations
        viewing_station.append(find_ground_station(gmst[k], x_true[:, k+1]))
        if viewing_station[k] != "none":
            # print(f"In view of {viewing_station[k]}")
            # z_k is [range, range_rate, El, Az] This needs noise added based on measurements. 
            meas_noise = np.random.multivariate_normal(np.zeros(4), R_k)
            z_k[:, k] = compute_measurement(gmst[k], viewing_station[k], x_true[:, k+1]) + meas_noise
            # print("True measurement: ", z_k[:, k])
            # if target is within viewing station complete update step of EKF 
            H = compute_H(gmst[k], viewing_station[k], xmkm[:, k+1])  # (4 x 6)
            H_k.append(H)

            # compute mapped measurement for z_hat_k
            z_hat = compute_measurement(gmst[k], viewing_station[k], xmkm[:, k+1])
            z_hat_k[:, k] = z_hat
            # print("mapped Measurement: ", z_hat)
            # compute innovations and cross covariance
            W = H @ Pmkm[k+1] @ H.T + R_k        
            C = Pmkm[k+1] @ H.T        
            W_k.append(W)
            C_k.append(C)

            # compute kalman gain
            K = np.linalg.solve(W.T, C.T).T    
            K_k.append(K)

            # update mean
            innovation = z_k[:, k] - z_hat
            innovation[3] = wrap_angle(innovation[3])  # azimuth (critical)  
            xmkp[:, k+1] = xmkm[:, k+1] + K @ innovation

            # update covariance use joesph stabilized form
            I = np.eye(nx)
            IKH = I - K @ H                       # (n x n)
            Pmkp_k = IKH @ Pmkm[k+1] @ IKH.T + K @ R_k @ K.T
            Pmkp_k = 0.5 * (Pmkp_k + Pmkp_k.T) + np.eye(6) * 1e-12
            Pmkp.append(Pmkp_k)
            # print("innovation norm:", np.linalg.norm(innovation))
            # print("K norm:", np.linalg.norm(K))

        else:
            # if target is outside viewing cone
            xmkp[:, k+1] = xmkm[:, k+1]
            Pmkp.append(Pmkm[k+1])

            # make measurements 0
            z_k[:, k] = np.zeros(nz)
            z_hat_k[:, k] = np.zeros(nz)
            H_k.append(np.zeros((nz, nx)))
            W_k.append(np.zeros((nz, nz)))
            C_k.append(np.zeros((nx, nz)))
            K_k.append(np.zeros((nx, nz)))

    plot_error(tf, x_true, xmkm, Pmkm)
    plot_error_components(tf, x_true, xmkm, Pmkm)
    plot_viewing(tf, viewing_station)
    plot_trajectory(x_true, xmkm)    
    # plot_trajectory_gs(x_true, xmkm, config.ground_stations)    
    plot_groundtrack(tf, x_true, xmkm, viewing_station, config.ground_stations, dt)
    plt.show()

    return True
def plot_viewing(tf, viewing_stations):
    N = len(viewing_stations)
    t = np.linspace(0, tf, N)

    unique_stations = [s for s in dict.fromkeys(viewing_stations) if s != "none"]
    cmap = plt.get_cmap("tab10")
    color_map = {station: cmap(i) for i, station in enumerate(unique_stations)}

    fig, ax = plt.subplots(figsize=(12, 3))

    i = 0
    while i < N:
        station = viewing_stations[i]
        if station != "none":
            j = i + 1
            while j < N and viewing_stations[j] == station:
                j += 1
            ax.axvspan(t[i], t[j - 1], alpha=0.5, color=color_map[station], label=station)
            i = j
        else:
            i += 1

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(), unique.keys(),
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        title="Ground Stations"
    )

    ax.set_yticks([])
    ax.set_xlabel("Time [s]")
    ax.set_title("Ground Station Visibility")
    ax.grid(True, axis="x")
    plt.tight_layout()


def plot_error(tf, x_true, x_noise, Pmkm):
    Pmkm = np.array(Pmkm)

    if Pmkm.shape[0] != 6:
        Pmkm = Pmkm.transpose(1, 2, 0)

    N = x_true.shape[1]
    t = np.linspace(0, tf, N) / 3600
    x_error = x_noise - x_true

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Position Error Norm (Indices 0, 1, 2) ---
    pos_error_norm = np.linalg.norm(x_error[0:3, :], axis=0)
    pos_sigma_3 = 3 * np.sqrt(Pmkm[0, 0, :] + Pmkm[1, 1, :] + Pmkm[2, 2, :])

    ax[0].plot(t, pos_error_norm, label=r"$\|\delta \mathbf{r}\|$")
    ax[0].plot(t, pos_sigma_3, 'k--', alpha=0.5, label=r"$3\sigma$")
    ax[0].set_title("Position Error Norm")
    ax[0].set_ylabel("Error [km]")
    ax[0].grid(True)
    ax[0].legend(loc='upper right')

    # --- Velocity Error Norm (Indices 3, 4, 5) ---
    vel_error_norm = np.linalg.norm(x_error[3:6, :], axis=0)
    vel_sigma_3 = 3 * np.sqrt(Pmkm[3, 3, :] + Pmkm[4, 4, :] + Pmkm[5, 5, :])

    ax[1].plot(t, vel_error_norm, label=r"$\|\delta \mathbf{v}\|$")
    ax[1].plot(t, vel_sigma_3, 'k--', alpha=0.5, label=r"$3\sigma$")
    ax[1].set_title("Velocity Error Norm")
    ax[1].set_ylabel("Error [km/s]")
    ax[1].set_xlabel("Time [hr]")
    ax[1].grid(True)
    ax[1].legend(loc='upper right')

    plt.tight_layout()

def plot_error_components(tf, x_true, x_noise, Pmkm):
    Pmkm = np.array(Pmkm)

    if Pmkm.shape[0] != 6:
        Pmkm = Pmkm.transpose(1, 2, 0)

    N = x_true.shape[1]
    t = np.linspace(0, tf, N) / 3600
    x_error = x_noise - x_true

    labels_r = [r"$\delta r_x$", r"$\delta r_y$", r"$\delta r_z$"]
    labels_v = [r"$\delta v_x$", r"$\delta v_y$", r"$\delta v_z$"]

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    # --- Position Error (row 0) ---
    for i in range(3):
        sigma_3 = 3 * np.sqrt(Pmkm[i, i, :])

        ax[0, i].plot(t, x_error[i, :], label=labels_r[i])
        ax[0, i].plot(t,  sigma_3, 'k--', alpha=0.5, label=r"$\pm 3\sigma$")
        ax[0, i].plot(t, -sigma_3, 'k--', alpha=0.5)
        ax[0, i].fill_between(t, -sigma_3, sigma_3, color='gray', alpha=0.15)

        ax[0, i].set_title(f"Position Error {labels_r[i]}")
        ax[0, i].set_ylabel("Error [km]")
        ax[0, i].legend(loc='upper right')
        ax[0, i].grid(True)

    # --- Velocity Error (row 1) ---
    for i in range(3):
        idx = i + 3
        sigma_3 = 3 * np.sqrt(Pmkm[idx, idx, :])

        ax[1, i].plot(t, x_error[idx, :], label=labels_v[i])
        ax[1, i].plot(t,  sigma_3, 'k--', alpha=0.5, label=r"$\pm 3\sigma$")
        ax[1, i].plot(t, -sigma_3, 'k--', alpha=0.5)
        ax[1, i].fill_between(t, -sigma_3, sigma_3, color='gray', alpha=0.15)

        ax[1, i].set_title(f"Velocity Error {labels_v[i]}")
        ax[1, i].set_ylabel("Error [km/s]")
        ax[1, i].set_xlabel("Time [hr]")
        ax[1, i].legend(loc='upper right')
        ax[1, i].grid(True)

    fig.suptitle("Position and Velocity Error Components", fontsize=14)
    plt.tight_layout()


def plot_trajectory(x_true, x_est):
    img = np.array(Image.open("earth.jpg").convert("RGB").resize((512*2, 256*2))) / 255.0

    u = np.linspace(0, 2 * np.pi, img.shape[1] + 1)
    v = np.linspace(0, np.pi,     img.shape[0] + 1)
    lon, lat = np.meshgrid(u, v)
    x = R_EARTH * np.cos(lon) * np.sin(lat)
    y = R_EARTH * np.sin(lon) * np.sin(lat)
    z = R_EARTH * np.cos(lat)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, facecolors=img, rstride=4, cstride=4, shade=False)
    ax.plot(x_true[0, :], x_true[1, :], x_true[2, :], color='blue', linewidth=1.5, zorder=5, label="True")
    # ax.plot(x_est[0, :], x_est[1, :], x_est[2, :], color='red', linewidth=1.5, zorder=5, label="Estimation")
    # Force equal aspect so a circular orbit looks circular
    orbit_max = np.max(np.abs(x_true[:3, :]))
    lim = orbit_max * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    plt.tight_layout()

def plot_groundtrack(tf, x_true, x_est, viewing_stations, ground_stations, dt):
    N = x_true.shape[1]
    t_sim = np.linspace(0, tf, N)

    # --- Convert ECI to lat/lon accounting for Earth's rotation ---
    t0 = datetime(2026, 4, 30, 0, 0, 0, tzinfo=timezone.utc)
    lat_true = np.zeros(N)
    lon_true = np.zeros(N)
    lat_est  = np.zeros(N)
    lon_est  = np.zeros(N)

    for k in range(N):
        tgmst  = t0 + timedelta(seconds=k * dt)
        gmst_k = gmst_from_datetime(tgmst)

        # True
        r_ecf_true = eci_to_ecef(x_true[:3, k], gmst_k)
        lat_true[k] = np.degrees(np.arcsin(r_ecf_true[2] / np.linalg.norm(r_ecf_true)))
        lon_true[k] = np.degrees(np.arctan2(r_ecf_true[1], r_ecf_true[0]))

        # Estimated
        r_ecf_est = eci_to_ecef(x_est[:3, k], gmst_k)
        lat_est[k] = np.degrees(np.arcsin(r_ecf_est[2] / np.linalg.norm(r_ecf_est)))
        lon_est[k] = np.degrees(np.arctan2(r_ecf_est[1], r_ecf_est[0]))
    # --- Ground station lat/lon ---
    cmap = plt.get_cmap("tab10")

    # --- Plot ---
    img = np.array(Image.open("earth.jpg").convert("RGB"))
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(img, extent=[-180, 180, -90, 90], aspect='auto', zorder=0)

    # Break ground track at anti-meridian crossings to avoid wrap-around lines
    def split_at_antimeridian(lon, lat):
        segments_lon, segments_lat = [], []
        seg_lon, seg_lat = [lon[0]], [lat[0]]
        for i in range(1, len(lon)):
            if abs(lon[i] - lon[i-1]) > 180:
                segments_lon.append(seg_lon)
                segments_lat.append(seg_lat)
                seg_lon, seg_lat = [], []
            seg_lon.append(lon[i])
            seg_lat.append(lat[i])
        segments_lon.append(seg_lon)
        segments_lat.append(seg_lat)
        return segments_lon, segments_lat

    segs_lon_true, segs_lat_true = split_at_antimeridian(lon_true, lat_true)
    segs_lon_est,  segs_lat_est  = split_at_antimeridian(lon_est,  lat_est)

    for s, (sl, slat) in enumerate(zip(segs_lon_true, segs_lat_true)):
        ax.plot(sl, slat, color='cyan', linewidth=0.8, zorder=2,
                label="True" if s == 0 else None)

    for s, (sl, slat) in enumerate(zip(segs_lon_est, segs_lat_est)):
        ax.plot(sl, slat, color='red', linewidth=0.8, linestyle='--', zorder=2,
                label="Estimated" if s == 0 else None)

    # Mark start point
    ax.scatter(lon_true[0], lat_true[0], color='white', s=60, zorder=5, marker='*', label="Start")

    # --- Colour-coded viewing windows along the track ---
    unique_stations = [s for s in dict.fromkeys(viewing_stations) if s != "none"]
    gs_cmap = {name: cmap(i) for i, name in enumerate(unique_stations)}

    for k, station in enumerate(viewing_stations):
        if station != "none":
            ax.scatter(lon_true[k], lat_true[k],
                       color=gs_cmap[station], s=6, zorder=3)

    # --- Ground station markers ---
    for i, (name, gs) in enumerate(ground_stations.items()):
        gs_lat = np.degrees(gs["lat"])
        gs_lon = np.degrees(gs["lon"])
        ax.scatter(gs_lon, gs_lat,
                   color=cmap(i), s=120, marker='^',
                   edgecolors='white', linewidths=0.6,
                   zorder=6, label=name)
        ax.text(gs_lon + 1.5, gs_lat + 1.5, name,
                fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.4),
                zorder=7)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Ground Track")
    ax.grid(True, color='white', alpha=0.2, zorder=1)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=8,
        title="Legend"
    )
    plt.tight_layout()

def plot_trajectory_gs(x_true, x_est, ground_stations):
    img = np.array(Image.open("earth.jpg").convert("RGB").resize((512*2, 256*2))) / 255.0

    u = np.linspace(0, 2 * np.pi, img.shape[1] + 1)
    v = np.linspace(0, np.pi,     img.shape[0] + 1)
    lon, lat = np.meshgrid(u, v)
    x = R_EARTH * np.cos(lon) * np.sin(lat)
    y = R_EARTH * np.sin(lon) * np.sin(lat)
    z = R_EARTH * np.cos(lat)

    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, facecolors=img, rstride=4, cstride=4, shade=False)
    ax.plot(x_true[0, :], x_true[1, :], x_true[2, :], color='blue', linewidth=1.5, zorder=5, label="True")
    # ax.plot(x_est[0, :], x_est[1, :], x_est[2, :], color='red', linewidth=1.5, zorder=5, label="Estimation")

    # --- Ground Stations ---
    for i, (name, gs) in enumerate(ground_stations.items()):
        gx, gy, gz = gs["r_ecf"]
        ax.scatter(gx, gy, gz,
                   color=cmap(i),
                   s=80,
                   zorder=10,
                   label=name)
        ax.text(gx * 1.05, gy * 1.05, gz * 1.05,
                name,
                fontsize=7,
                color=cmap(i),
                zorder=10)

    orbit_max = np.max(np.abs(x_true[:3, :]))
    lim = orbit_max * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=8
    )

    plt.tight_layout()
    


if __name__ == "__main__":
    main()