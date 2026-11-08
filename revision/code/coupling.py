from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm, expm_frechet

from paths import output
OUT = output('coupling')
pauli = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]], complex)
S = np.array([np.kron(x, np.eye(2)) for x in pauli])
ZZ = np.kron(pauli[2], pauli[2])
phi = np.array([1,0,0,1], complex) / np.sqrt(2)
bell = np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1,-1,0]], complex) / np.sqrt(2)
M = bell @ expm(-1j*np.pi/(3*np.sqrt(3))*sum(S))

def fisher(J, T):
    A = -1j*T*J*ZZ
    psi = expm(A) @ phi
    ds = np.array([expm_frechet(A, -1j*T*x, compute_expm=False) @ phi for x in S])
    amps = M @ psi
    dp = 2*np.real((ds @ M.T)*amps.conj())
    W = dp / np.abs(amps)
    F = W @ W.T
    ov = ds.conj() @ psi
    Q = 4*np.real(ds.conj() @ ds.T - np.outer(ov, ov.conj()))
    return psi, ds, F, Q

checks = []
for J,T in [(0.,8.), (.01,10.), (.1,2.), (.2,8.), (.5,np.pi), (1.,2.), (10.,6.)]:
    psi, ds, F, Q = fisher(J,T)
    ft = (2*T*np.sinc(2*J*T/np.pi))**2
    qt = (2*T*np.sinc(J*T/np.pi))**2
    Fa = np.diag([ft,ft,4*T*T])
    Qa = np.diag([qt,qt,4*T*T])
    ferr = float(np.linalg.norm(F-Fa)/np.linalg.norm(Fa))
    qerr = float(np.linalg.norm(Q-Qa)/np.linalg.norm(Qa))
    assert max(ferr,qerr)<2e-13
    assert np.linalg.norm(psi-np.exp(-1j*J*T)*phi)<2e-14
    checks.append(dict(J=J,T=T,CFIM_relative_error=ferr,QFIM_relative_error=qerr))

J,T=.2,8.
psi,ds,F,Q=fisher(J,T)
B=1.;theta=azimuth=np.pi/4
Jac=np.array([[np.sin(theta)*np.cos(azimuth),B*np.cos(theta)*np.cos(azimuth),-B*np.sin(theta)*np.sin(azimuth)],
              [np.sin(theta)*np.sin(azimuth),B*np.cos(theta)*np.sin(azimuth),B*np.sin(theta)*np.cos(azimuth)],
              [np.cos(theta),-B*np.sin(theta),0.]])
CF=float(np.trace(np.linalg.inv(Jac.T@F@Jac)))
CQ=float(np.trace(np.linalg.inv(Jac.T@Q@Jac)))
h=1e-6
dpfd=[]
for S_i in S:
    pp=abs(M@expm(-1j*T*(J*ZZ+h*S_i))@phi)**2
    pm=abs(M@expm(-1j*T*(J*ZZ-h*S_i))@phi)**2
    dpfd.append((pp-pm)/(2*h))
amps=M@psi
Ffd=(np.array(dpfd)/abs(amps))@(np.array(dpfd)/abs(amps)).T
fd_error=float(np.linalg.norm(F-Ffd)/np.linalg.norm(F))
assert fd_error<1e-7

# Directly verify the conserved-sector decomposition without using it to build H.
hs=.37*pauli[0]-.21*pauli[1]+.53*pauli[2]
H=np.kron(hs,np.eye(2))+.4*ZZ
U=expm(-1j*.71*H)
Up=expm(-1j*.71*(hs+.4*pauli[2]));Um=expm(-1j*.71*(hs-.4*pauli[2]))
sector_error=float(np.linalg.norm(U-np.kron(Up,np.diag([1,0]))-np.kron(Um,np.diag([0,1]))))
assert sector_error<2e-14
z_commutator=float(np.linalg.norm(S[2]@ZZ-ZZ@S[2]))
assert z_commutator==0

data=dict(method='Independent 4x4 matrix exponential and Frechet derivatives; controls and POVM fixed.',
          scope='Analytical unoptimized-control results for the static field.',
          checks=checks,static_example=dict(J=J,T=T,B=B,theta=theta,phi=azimuth,
          F_cartesian=F.tolist(),Q_cartesian=Q.tolist(),C_fixed_measurement=CF,C_SLD=CQ,
          C_J0=1/T**2,finite_difference_relative_error=fd_error),
          sector_decomposition_error=sector_error,z_rotation_Ising_commutator_norm=z_commutator,
          new_optimization_runs=0)
(OUT/'analytic_verification.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(dict(C_fixed_measurement=CF,C_SLD=CQ,max_formula_relative_error=max(max(r['CFIM_relative_error'],r['QFIM_relative_error']) for r in checks),finite_difference_relative_error=fd_error,new_optimization_runs=0)))
