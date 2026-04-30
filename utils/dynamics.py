import numpy as np
from config import MU_EARTH, R_EARTH, J2
from utils.dynamics_jacobian import dynamics_jacobian

def spacecraft_dyn(t, x, w):
    r = x[0:3]
    v = x[3:6]

    r_norm = np.linalg.norm(r)

    a = ((-MU_EARTH * r) / r_norm**3 + j2_acceleration(r)).flatten()

    x_dot = np.hstack((v, a))
    return x_dot

def j2_acceleration(r_vec):
    x, y, z = r_vec
    Re = R_EARTH
    r = np.linalg.norm(r_vec)
    
    factor = (1.5 * J2 * MU_EARTH * Re**2) / r**5
    zx2 = 5 * z**2 / r**2
    
    ax = factor * x * (zx2 - 1)
    ay = factor * y * (zx2 - 1)
    az = factor * z * (zx2 - 3)
    
    return np.array([ax, ay, az])

def spacecraft_dyn_stm(t, x_aug, G, Q):
    # unpack state
    x = x_aug[0:6]
    r = x_aug[0:3]
    v = x_aug[3:6]
    P = x_aug[6:(6 + 6**2)].reshape(6, 6)

    # Dynamics
    r_norm = np.linalg.norm(r)
    a = (-MU_EARTH * r) / r_norm**3
    x_dot = np.hstack((v, a)) # spacecraft_dyn(t, x, np.zeros(3))
    A = dynamics_jacobian(x, MU_EARTH, J2, R_EARTH)
    P_dot = A @ P + P @ A.T + G @ Q @ G.T

    # pack state
    x_aug_dot = np.concatenate([x_dot, P_dot.flatten()])
    return x_aug_dot
