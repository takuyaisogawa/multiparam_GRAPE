import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. Global constants & Pauli operators
# ──────────────────────────────────────────────
PI        = np.pi
B0     = 1.0
theta0 = PI / 4
phi0   = PI / 4
h_FD      = 1.0e-4
theta_rot = PI / 3
total_T   = 2.0
N_slices  = 50
tau       = total_T / N_slices
n_epochs  = 100
J         = 0.01

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)

sigma1x = np.kron(sigma_x, I2)
sigma1y = np.kron(sigma_y, I2)
sigma1z = np.kron(sigma_z, I2)
I4 = np.eye(4, dtype=np.complex128)

# Bell-state projectors
bell_states = np.array([
    [1,  0,  0, -1],
    [0,  1,  1,  0],
    [0,  1, -1,  0],
    [1,  0,  0,  1]
], dtype=np.complex128) / np.sqrt(2)

projectors = np.empty((4, 4, 4), dtype=np.complex128)
for k in range(4):
    v = bell_states[k]
    projectors[k] = np.outer(v, v.conj())

psi0 = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
rho0 = np.outer(psi0, psi0.conj())

# ──────────────────────────────────────────────
# 2. Unitary evolution with full Hamiltonian
# ──────────────────────────────────────────────
@njit(inline="always")
def U4_full(vx, vy, vz, Jcouple, dt):
    hz = np.array([vz + Jcouple, vz - Jcouple])
    U = np.zeros((4, 4), dtype=np.complex128)

    for blk in range(2):
        mag = np.sqrt(vx*vx + vy*vy + hz[blk]*hz[blk])
        if mag < 1e-14:
            cs = 1.0
            ss = 0.0
        else:
            cs = np.cos(mag * dt)
            ss = np.sin(mag * dt) / mag

        a = cs - 1j * ss * hz[blk]
        d = cs + 1j * ss * hz[blk]
        b = -1j * ss * (vx - 1j*vy)
        c = -1j * ss * (vx + 1j*vy)

        if blk == 0:
            U[0, 0] = a; U[0, 2] = b
            U[2, 0] = c; U[2, 2] = d
        else:
            U[1, 1] = a; U[1, 3] = b
            U[3, 1] = c; U[3, 3] = d
    return U

@njit
def probabilities(theta_rot, tau, n_slices,
                  B, theta, phi,
                  Vx, Vy, Vz, Jcouple,
                  projectors):
    rho = rho0.copy()
    Bx = B * np.sin(theta) * np.cos(phi)
    By = B * np.sin(theta) * np.sin(phi)
    Bz = B * np.cos(theta)

    for t in range(n_slices):
        U = U4_full(Bx + Vx[t], By + Vy[t], Bz + Vz[t], Jcouple, tau)
        rho = U @ rho @ U.conj().T

    coeff = 1.0 / np.sqrt(3.0)
    U_rot = U4_full(coeff, coeff, coeff, 0.0, theta_rot)
    rho = U_rot @ rho @ U_rot.conj().T

    p = np.empty(4, dtype=np.float64)
    for k in range(4):
        p[k] = np.real(np.trace(rho @ projectors[k]))
    return p

@njit
def f0_and_trace(Vx, Vy, Vz,
                 theta_rot, tau, n_slices,
                 B0, theta0, phi0,
                 h, Jcouple, projectors):
    P0   = probabilities(theta_rot, tau, n_slices, B0,   theta0,   phi0,   Vx, Vy, Vz, Jcouple, projectors)
    P_Bp = probabilities(theta_rot, tau, n_slices, B0+h, theta0,   phi0,   Vx, Vy, Vz, Jcouple, projectors)
    P_Bm = probabilities(theta_rot, tau, n_slices, B0-h, theta0,   phi0,   Vx, Vy, Vz, Jcouple, projectors)
    P_tp = probabilities(theta_rot, tau, n_slices, B0,   theta0+h, phi0,   Vx, Vy, Vz, Jcouple, projectors)
    P_tm = probabilities(theta_rot, tau, n_slices, B0,   theta0-h, phi0,   Vx, Vy, Vz, Jcouple, projectors)
    P_pp = probabilities(theta_rot, tau, n_slices, B0,   theta0,   phi0+h, Vx, Vy, Vz, Jcouple, projectors)
    P_pm = probabilities(theta_rot, tau, n_slices, B0,   theta0,   phi0-h, Vx, Vy, Vz, Jcouple, projectors)

    grad = np.empty((3, 4))
    grad[0] = (P_Bp - P_Bm) / (2 * h)
    grad[1] = (P_tp - P_tm) / (2 * h)
    grad[2] = (P_pp - P_pm) / (2 * h)

    F = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(4):
                acc += grad[i, k] * grad[j, k] / P0[k]
            F[i, j] = acc

    Cov = np.linalg.inv(F)
    tr_inv = Cov[0, 0] + Cov[1, 1] + Cov[2, 2]
    f0 = 1.0 / (1.0 / F[0, 0] + 1.0 / F[1, 1] + 1.0 / F[2, 2])
    return f0, tr_inv


@njit(cache=True)
def literal_step(V1x,V1y,V1z,J,tau,epoch):
    if epoch <= 300:
        lr = 1e-08
    else:
        lr = 1e-06
    f0_base, _ = f0_and_trace(V1x, V1y, V1z, theta_rot, tau, N_slices, B0, theta0, phi0, h_FD, J, projectors)
    grads = []
    eps = 1e-05
    for V in (V1x, V1y, V1z):
        g = np.zeros_like(V)
        for i in range(N_slices):
            V[i] += eps
            f_pert, _ = f0_and_trace(V1x, V1y, V1z, theta_rot, tau, N_slices, B0, theta0, phi0, h_FD, J, projectors)
            g[i] = (f_pert - f0_base) / eps
            V[i] -= eps
        grads.append(g)
    V1x += lr * grads[0]
    V1y += lr * grads[1]
    V1z += lr * grads[2]
    return grads[0], grads[1], grads[2], lr
