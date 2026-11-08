import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# ─────────────────────────────────────
# 1. Global constants & Pauli operators
# ─────────────────────────────────────
PI        = np.pi
B0     = 1.0
omega0 = 1.0
h_FD      = 1.0e-4
total_T   = 2.0
N_slices  = 500
tau       = total_T / N_slices
theta_rot = PI / 3
n_epochs  = 100
J         = 0.01

# Pauli matrices
σx = np.array([[0,1],[1,0]],complex)
σy = np.array([[0,-1j],[1j,0]],complex)
σz = np.array([[1,0],[0,-1]],complex)
I2 = np.eye(2, dtype=complex)

σ1x,σ1y,σ1z = np.kron(σx,I2),np.kron(σy,I2),np.kron(σz,I2)
ZZ          = np.kron(σz,σz)

# Bell-state projectors
bell = np.array([[1,0,0,-1],
                 [0,1,1,0],
                 [0,1,-1,0],
                 [1,0,0,1]],complex)/np.sqrt(2)
projs = np.array([np.outer(v,v.conj()) for v in bell])

ψ0  = np.array([1,0,0,1],complex)/np.sqrt(2)
ρ0  = np.outer(ψ0,ψ0.conj())

# ─────────────────────────────────────
# 2. Unitary block -— target = −Bcos σx − Bsin σz + JZZ
# ─────────────────────────────────────
@njit(inline='always')
def U4_full(vx, vy, vz, Jcouple, dt):
    hz = np.array([vz+Jcouple, vz-Jcouple])
    U  = np.zeros((4,4),dtype=np.complex128)
    for blk in range(2):
        mag = np.sqrt(vx*vx + vy*vy + hz[blk]*hz[blk])
        if mag < 1e-14:
            cs=1.0; ss=0.0
        else:
            cs = np.cos(mag*dt)
            ss = np.sin(mag*dt)/mag
        a = cs-1j*ss*hz[blk]; d = cs+1j*ss*hz[blk]
        b = -1j*ss*(vx-1j*vy); c = -1j*ss*(vx+1j*vy)
        if blk==0:
            U[0,0]=a; U[0,2]=b; U[2,0]=c; U[2,2]=d
        else:
            U[1,1]=a; U[1,3]=b; U[3,1]=c; U[3,3]=d
    return U

@njit
def probabilities(theta_rot, tau, n_slices,
                  B, ω,
                  Vx, Vy, Vz, Jcouple,
                  projs):
    ρ = ρ0.copy()
    for t in range(n_slices):
        t_now = t*tau
        # --- target fields ---
        vx_t = -B * np.cos(ω*t_now)
        vy_t = 0.0
        vz_t = -B * np.sin(ω*t_now)
        # --- control fields ---
        vx_t += Vx[t]; vy_t += Vy[t]; vz_t += Vz[t]
        U  = U4_full(vx_t, vy_t, vz_t, Jcouple, tau)
        ρ  = U @ ρ @ U.conj().T

    # measurement-axis rotation
    c = 1/np.sqrt(3)
    Urot = U4_full(c,c,c,0.0,theta_rot)
    ρ = Urot @ ρ @ Urot.conj().T

    p = np.empty(4)
    for k in range(4):
        p[k] = np.real(np.trace(ρ @ projs[k]))
    return p

@njit
def f0_and_trace(Vx,Vy,Vz,
                 theta_rot,tau,n_slice,
                 B0,ω0,h,Jcouple,projs):
    # base + ±h on B and ω
    P0   = probabilities(theta_rot,tau,n_slice,B0,   ω0,   Vx,Vy,Vz,Jcouple,projs)
    PBp  = probabilities(theta_rot,tau,n_slice,B0+h, ω0,   Vx,Vy,Vz,Jcouple,projs)
    PBm  = probabilities(theta_rot,tau,n_slice,B0-h, ω0,   Vx,Vy,Vz,Jcouple,projs)
    Pwp  = probabilities(theta_rot,tau,n_slice,B0,   ω0+h,Vx,Vy,Vz,Jcouple,projs)
    Pwm  = probabilities(theta_rot,tau,n_slice,B0,   ω0-h,Vx,Vy,Vz,Jcouple,projs)

    grad = np.empty((2,4))
    grad[0]=(PBp-PBm)/(2*h)
    grad[1]=(Pwp-Pwm)/(2*h)

    F=np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            acc=0.0
            for k in range(4):
                acc += grad[i,k]*grad[j,k]/max(P0[k],1e-12)
            F[i,j]=acc
    Cov = np.linalg.inv(F)
    tr_inv = Cov[0,0]+Cov[1,1]
    f0 = 1.0/(1/F[0,0]+1/F[1,1])
    return f0,tr_inv
