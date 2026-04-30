import sympy as sp

# Constants
mu, J2, r_earth = sp.symbols("mu J2 r_earth")

# State Symbols
rx, ry, rz = sp.symbols("rx ry rz")
vx, vy, vz = sp.symbols("vx, vy, vz")

r = sp.Matrix([rx, ry, rz])
v = sp.Matrix([vx, vy, vz])

x = sp.Matrix([r, v]) # State vector

# Position Derivative
r_dot = v

# Velocity Derivative
r_norm = sp.sqrt(r.dot(r))
a = -(mu / r_norm**3) * r
factor = (1.5 * J2 * mu * r_earth**2) / r_norm**5
zx2 = 5 * rz**2 / r_norm**2
ax_j2 = factor * rx * (zx2 - 1)
ay_j2 = factor * ry * (zx2 - 1)
az_j2 = factor * rz * (zx2 - 3)
a_j2 = sp.Matrix([ax_j2, ay_j2, az_j2])
v_dot = a # + a_j2

# Full dynamics
f = sp.Matrix.vstack(r_dot, v_dot)

A = f.jacobian(x) # Take jacobian w.r.t state vector

inputs = (
    rx, ry, rz,
    vx, vy, vz,
    mu, J2, r_earth
)

A_func = sp.lambdify(inputs, A, 'numpy')

def dynamics_jacobian(x, mu, J2, r_earth):
    return A_func(
        x[0], x[1], x[2],
        x[3], x[4], x[5],
        mu, J2, r_earth
    )

