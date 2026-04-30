import numpy as np
from config import MU_EARTH, R_EARTH, x_0, P0, Q_s, R_k
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from utils.dynamics import spacecraft_dyn, spacecraft_dyn_stm
from utils.frame_conversion import *
from utils.measurement import *



# TODO: 
# 1. dynamics function w/ stm
# 2. jacobian using sympy
# 3. stats initialization
# 4. test functions by propagating true state and plot with a rotating earth

np.random.seed(0)

def main():
    tf = 1 * 24 * 60 * 60 # time -> seconds
    dt = 60 # seconds
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
    xmkm = np.zeros((nx, N+1)) # TODO: Add some noise. It needs to match what is added to the xmkp
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
        gmst[k] = get_gmst(tkp1)

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
            np.concatenate([xmkm[:, k], Pmkp[k].flatten()]),
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
            z_k[:, k] = compute_measurement(gmst, viewing_station[k], x_true)

            # if target is within viewing station complete update step of EKF 
            # compute mapped measurement for z_hat_k
            # compute innovations and cross covariance
            # compute kalman gain
            # update mean
            # update covariance use joesph stabilized form
        else:
            # if target is outside viewing cone
            xmkp[:, k+1] = xmkm[:, k+1]
            Pmkp.append(Pmkm[k+1])

    plot_error(tf, x_true, xmkm)
    plot_viewing(tf, viewing_station)
    plot_trajectory(x_true, xmkm)    
    plt.show()

    return True

def plot_viewing(tf, viewing_stations):
    N = len(viewing_stations)
    view = np.zeros(N)
    for k in range(N):
        if viewing_stations[k] != "none":
            view[k] = 1

    t = np.linspace(0, tf, N)
    fig, ax = plt.subplots()
    ax.plot(t, view)
    ax.grid()
    ax.set_title("Viewing Time")

def plot_error(tf, x_true, x_noise):
    N = x_true.shape[1]
    t = np.linspace(0, tf, N)
    x_error = x_noise - x_true
    fig, ax = plt.subplots(2,1)
    ax[0].plot(t, x_error[0, :], label=r"$\delta r_x$")
    ax[0].plot(t, x_error[1, :], label=r"$\delta r_y$")
    ax[0].plot(t, x_error[2, :], label=r"$\delta r_z$")
    ax[0].set_title("Position Error")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, x_error[3, :], label=r"$\delta v_x$")
    ax[1].plot(t, x_error[3, :], label=r"$\delta v_y$")
    ax[1].plot(t, x_error[3, :], label=r"$\delta v_z$")
    ax[1].set_title("Velocity Error")
    ax[1].grid()
    ax[1].legend()

    plt.tight_layout()

def plot_trajectory(x_true, x_est):
    img = np.array(Image.open("earth.jpg").convert("RGB").resize((512, 256))) / 255.0

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
    ax.plot(x_est[0, :], x_est[1, :], x_est[2, :], color='red', linewidth=1.5, zorder=5, label="Estimation")
    # Force equal aspect so a circular orbit looks circular
    orbit_max = np.max(np.abs(x_true[:3, :]))
    lim = orbit_max * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    plt.tight_layout()
    


if __name__ == "__main__":
    main()