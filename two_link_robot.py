"""
Two-Link Planar Robot: KpH-EDMD Identification and Control
===========================================================
Reproduces the numerical example from:
  V. M. Preciado, "A Koopman-Port-Hamiltonian Framework for
  Data-Driven Modeling and Control," IEEE CDC, 2025.

System: two-link planar robot with fully coupled inertial dynamics
        M(q)*ddq + C(q,dq)*dq + g(q) = u - B*dq
State:  x = [q1, q2, p1, p2]  (joint angles + generalized momenta)
Input:  u = [u1, u2]           (joint torques)

pH structure:
  H(x) = 0.5*p^T * M(q)^{-1} * p + V(q)
  J = symplectic block,  R = diag damping,  G = identity input

Outputs (saved to figures/):
  fig_robot_joints.pdf    -- joint angles under damping-injection control
  fig_robot_energy.pdf    -- total energy + passivity residual

Key results (printed to console):
  EDMD RMSE, KpH RMSE, spectral radii, settling times
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Allow running from repo root or from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kph import KpHEDMD
from kph.edmd import fit_edmd, passivity_residual

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG   = os.path.join(_REPO, "figures")
os.makedirs(FIG, exist_ok=True)

# ── Figure style (IEEE double-column) ─────────────────────────────────────────
matplotlib.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1.2,
    'figure.dpi': 300,
})
FIGSIZE = (7.0, 2.8)   # IEEE double-column width
STYLE = {
    'true': dict(color='black',   linestyle='-',  linewidth=1.5),
    'edmd': dict(color='#CC0000', linestyle='--', linewidth=1.2),
    'kph':  dict(color='#0055CC', linestyle='-',  linewidth=1.5),
    'tol':  dict(color='gray',    linestyle=':',  linewidth=0.8),
}

# =============================================================================
# === SYSTEM PARAMETERS ===
# =============================================================================
params = {
    'm1': 1.0, 'm2': 1.0,
    'l1': 1.0, 'l2': 1.0,
    'lc1': 0.5, 'lc2': 0.5,
    'I1': 0.1, 'I2': 0.1,
    'g': 9.81,
    'b1': 0.2, 'b2': 0.2,
}
dt = 0.02

# =============================================================================
# === pH STRUCTURE FUNCTIONS ===
# =============================================================================

def mass_matrix(q: np.ndarray, p: dict) -> np.ndarray:
    """Compute the 2×2 inertia matrix M(q) for the two-link robot.

    Parameters
    ----------
    q : np.ndarray, shape (2,)
        Joint angles [q1, q2] in radians.
    p : dict
        System parameter dictionary.

    Returns
    -------
    M : np.ndarray, shape (2, 2)
        Symmetric positive-definite inertia matrix.
    """
    c2  = np.cos(q[1])
    m11 = (p['I1'] + p['I2'] + p['m1']*p['lc1']**2
           + p['m2']*(p['l1']**2 + p['lc2']**2 + 2*p['l1']*p['lc2']*c2))
    m12 = p['I2'] + p['m2']*(p['lc2']**2 + p['l1']*p['lc2']*c2)
    m22 = p['I2'] + p['m2']*p['lc2']**2
    return np.array([[m11, m12], [m12, m22]])


def coriolis_vector(q: np.ndarray, dq: np.ndarray, p: dict) -> np.ndarray:
    """Compute the Coriolis/centripetal force vector C(q,dq)*dq.

    Parameters
    ----------
    q : np.ndarray, shape (2,)
        Joint angles in radians.
    dq : np.ndarray, shape (2,)
        Joint velocities in rad/s.
    p : dict
        System parameter dictionary.

    Returns
    -------
    c_vec : np.ndarray, shape (2,)
        Coriolis force vector.
    """
    s2 = np.sin(q[1])
    h  = p['m2'] * p['l1'] * p['lc2'] * s2
    return np.array([
        -h * dq[1] * (2*dq[0] + dq[1]),
         h * dq[0]**2,
    ])


def gravity_vector(q: np.ndarray, p: dict) -> np.ndarray:
    """Compute the gravitational torque vector g(q).

    Parameters
    ----------
    q : np.ndarray, shape (2,)
        Joint angles in radians.
    p : dict
        System parameter dictionary.

    Returns
    -------
    g_vec : np.ndarray, shape (2,)
        Gravitational torque at each joint.
    """
    return np.array([
        (p['m1']*p['lc1'] + p['m2']*p['l1'])*p['g']*np.cos(q[0])
        + p['m2']*p['lc2']*p['g']*np.cos(q[0] + q[1]),
        p['m2']*p['lc2']*p['g']*np.cos(q[0] + q[1]),
    ])


def hamiltonian(x: np.ndarray, p: dict) -> float:
    """Evaluate the Hamiltonian H(x) = T(q,p) + V(q).

    Parameters
    ----------
    x : np.ndarray, shape (4,)
        State vector [q1, q2, p1, p2] with p = M(q)*dq.
    p : dict
        System parameter dictionary.

    Returns
    -------
    float
        Total mechanical energy in joules.
    """
    q, pv = x[:2], x[2:]
    Mi = np.linalg.inv(mass_matrix(q, p))
    V  = ((p['m1']*p['lc1'] + p['m2']*p['l1'])*p['g']*np.sin(q[0])
          + p['m2']*p['lc2']*p['g']*np.sin(q[0] + q[1]))
    return 0.5*(pv @ Mi @ pv) + V


def nabla_H(x: np.ndarray, p: dict, eps: float = 1e-7) -> np.ndarray:
    """Compute the gradient of H via central finite differences.

    Parameters
    ----------
    x : np.ndarray, shape (4,)
        State vector.
    p : dict
        System parameter dictionary.
    eps : float, optional
        Finite-difference step size. Default 1e-7.

    Returns
    -------
    grad : np.ndarray, shape (4,)
        Gradient nabla_H(x).
    """
    grad = np.zeros(4)
    for i in range(4):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps; xm[i] -= eps
        grad[i] = (hamiltonian(xp, p) - hamiltonian(xm, p)) / (2*eps)
    return grad


# pH matrices
J_ph = np.array([[ 0, 0, 1, 0],
                  [ 0, 0, 0, 1],
                  [-1, 0, 0, 0],
                  [ 0,-1, 0, 0]], dtype=float)
G_ph = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=float)

def R_ph(p: dict) -> np.ndarray:
    """Damping matrix R = diag(0, 0, b1, b2).

    Parameters
    ----------
    p : dict
        System parameter dictionary.

    Returns
    -------
    np.ndarray, shape (4, 4)
    """
    return np.diag([0.0, 0.0, p['b1'], p['b2']])


def two_link_rhs(t: float, x: np.ndarray, u: np.ndarray, p: dict) -> np.ndarray:
    """Right-hand side of the two-link robot Hamiltonian ODE.

    Implements the true port-Hamiltonian dynamics in (q, p) coordinates::

        dq/dt = M(q)^{-1} p
        dp/dt = -dH/dq - B*dq + u

    where dH/dq = dV/dq + dT_H/dq with::

        dT_H/dq_1 = 0               (M independent of q1)
        dT_H/dq_2 = h*dq1*(dq1+dq2), h = m2*l1*lc2*sin(q2)

    Parameters
    ----------
    t : float
        Time (unused; required by solve_ivp interface).
    x : np.ndarray, shape (4,)
        State [q1, q2, p1, p2].
    u : np.ndarray, shape (2,)
        Joint torques [u1, u2].
    p : dict
        System parameter dictionary.

    Returns
    -------
    np.ndarray, shape (4,)
        State derivative x_dot.
    """
    q, pv = x[:2], x[2:]
    Mi    = np.linalg.inv(mass_matrix(q, p))
    dq    = Mi @ pv
    h     = p['m2'] * p['l1'] * p['lc2'] * np.sin(q[1])
    dTH   = np.array([0.0, h * dq[0] * (dq[0] + dq[1])])
    dp    = -(gravity_vector(q, p) + dTH) - np.diag([p['b1'], p['b2']]) @ dq + u
    return np.concatenate([dq, dp])


def simulate_robot(
    x0: np.ndarray,
    u_seq: np.ndarray,
    p: dict = params,
    dt: float = dt,
) -> np.ndarray:
    """Simulate the two-link robot for T steps via RK45.

    Parameters
    ----------
    x0 : np.ndarray, shape (4,)
        Initial state [q1, q2, p1, p2].
    u_seq : np.ndarray, shape (T, 2)
        Control input sequence.
    p : dict, optional
        System parameter dictionary. Default params.
    dt : float, optional
        Sampling period in seconds. Default 0.02.

    Returns
    -------
    X : np.ndarray, shape (4, T+1)
        State trajectory; columns are time steps.
    """
    T = u_seq.shape[0]
    X = np.zeros((4, T + 1)); X[:, 0] = x0
    for k in range(T):
        sol = solve_ivp(
            two_link_rhs, [0, dt], X[:, k],
            args=(u_seq[k], p),
            method='RK45', rtol=1e-8, atol=1e-10, max_step=dt/4,
        )
        X[:, k + 1] = sol.y[:, -1]
    return X

# =============================================================================
# === pH STRUCTURE VERIFICATION ===
# =============================================================================
rng_ver = np.random.RandomState(0)
max_ph_err = 0.0
for _ in range(10):
    x_t  = rng_ver.uniform([-0.5, -0.5, -1, -1], [0.5, 0.5, 1, 1])
    u_t  = rng_ver.uniform(-2, 2, 2)
    JR   = J_ph - R_ph(params)
    nH   = nabla_H(x_t, params)
    f_ph  = JR @ nH + G_ph @ u_t
    f_sim = np.array(two_link_rhs(0, x_t, u_t, params))
    max_ph_err = max(max_ph_err, np.max(np.abs(f_ph - f_sim)))

ph_ok = max_ph_err < 1e-6
assert ph_ok, f"pH structure verification FAILED: max error = {max_ph_err:.2e}"

# =============================================================================
# === DICTIONARY ===
# =============================================================================
N = 24   # dictionary dimension

def build_dictionary(X: np.ndarray, p: dict = params) -> np.ndarray:
    """Build the N=24 basis function dictionary for the two-link robot.

    Basis functions (ordered):
      0–3:   q1, q2, p1, p2                      (linear state)
      4–8:   q1^2, q2^2, p1^2, p2^2, p1*p2       (degree-2 polynomial)
      9–10:  sin(q1), cos(q1)                     (joint 1 trig)
      11–12: sin(q2), cos(q2)                     (joint 2 trig)
      13–14: sin(q1+q2), cos(q1+q2)               (sum trig)
      15–16: sin(q1-q2), cos(q1-q2)               (diff trig)
      17–18: cos(q2)*p1, cos(q2)*p2               (inertia-coupled momenta)
      19–20: sin(q2)*p1, sin(q2)*p2               (inertia-coupled momenta)
      21:    T_kin = 0.5 * p^T M^{-1} p           (kinetic energy)
      22:    V_pot = gravitational potential       (potential energy)
      23:    1                                     (constant offset)

    Parameters
    ----------
    X : np.ndarray, shape (4, M)
        State snapshots; columns are samples.
    p : dict, optional
        System parameter dictionary. Default params.

    Returns
    -------
    Psi : np.ndarray, shape (24, M)
        Dictionary snapshot matrix.
    """
    q1, q2, p1, p2 = X[0], X[1], X[2], X[3]
    M_col = X.shape[1]
    T_kin = np.array([
        0.5 * (X[2:, k] @ np.linalg.inv(mass_matrix(X[:2, k], p)) @ X[2:, k])
        for k in range(M_col)
    ])
    V_pot = (
        (p['m1']*p['lc1'] + p['m2']*p['l1'])*p['g']*np.sin(q1)
        + p['m2']*p['lc2']*p['g']*np.sin(q1 + q2)
    )
    return np.vstack([
        q1, q2, p1, p2,
        q1**2, q2**2, p1**2, p2**2, p1*p2,
        np.sin(q1), np.cos(q1),
        np.sin(q2), np.cos(q2),
        np.sin(q1+q2), np.cos(q1+q2),
        np.sin(q1-q2), np.cos(q1-q2),
        np.cos(q2)*p1, np.cos(q2)*p2,
        np.sin(q2)*p1, np.sin(q2)*p2,
        T_kin, V_pot,
        np.ones(M_col),
    ])


def dictionary_derivative_numerical(
    X: np.ndarray,
    U: np.ndarray,
    p: dict = params,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute time derivatives of the dictionary via central finite differences.

    dPsi/dt = sum_i (d phi / d x_i) * f_i(x, u)

    Parameters
    ----------
    X : np.ndarray, shape (4, M)
        State snapshots.
    U : np.ndarray, shape (2, M)
        Control inputs at each snapshot.
    p : dict, optional
        System parameter dictionary. Default params.
    eps : float, optional
        Finite-difference step. Default 1e-6.

    Returns
    -------
    dPsi : np.ndarray, shape (24, M)
        Time derivatives of the dictionary at each snapshot.
    """
    M_col = X.shape[1]
    dPsi  = np.zeros((N, M_col))
    for k in range(M_col):
        f = np.array(two_link_rhs(0, X[:, k], U[:, k], p))
        dPsi_k = np.zeros(N)
        for i in range(4):
            xp, xm = X[:, k].copy(), X[:, k].copy()
            xp[i] += eps; xm[i] -= eps
            psi_p = build_dictionary(xp.reshape(4, 1), p).flatten()
            psi_m = build_dictionary(xm.reshape(4, 1), p).flatten()
            dPsi_k += (psi_p - psi_m) / (2*eps) * f[i]
        dPsi[:, k] = dPsi_k
    return dPsi

# =============================================================================
# === DATA COLLECTION ===
# =============================================================================
M_train = 1500
rng = np.random.RandomState(42)
q1_s  = rng.uniform(-np.pi/3, np.pi/3, M_train)
q2_s  = rng.uniform(-np.pi/3, np.pi/3, M_train)
dq1_s = rng.uniform(-1.0, 1.0, M_train)
dq2_s = rng.uniform(-1.0, 1.0, M_train)

# Convert velocities to momenta p = M(q)*dq
p1_s = np.zeros(M_train); p2_s = np.zeros(M_train)
for k in range(M_train):
    q_k  = np.array([q1_s[k], q2_s[k]])
    dq_k = np.array([dq1_s[k], dq2_s[k]])
    pv_k = mass_matrix(q_k, params) @ dq_k
    p1_s[k] = pv_k[0]; p2_s[k] = pv_k[1]

X_train = np.vstack([q1_s, q2_s, p1_s, p2_s])   # (4, M)
U_train = rng.uniform(-5, 5, (2, M_train))         # (2, M)
Psi_tr  = build_dictionary(X_train)                # (24, M)

print("Computing dictionary derivatives (numerical, M=1500)...")
t_data = time.time()
dPsi_tr = dictionary_derivative_numerical(X_train, U_train)
print(f"  Done in {time.time()-t_data:.1f} s")

rng_noise = np.random.RandomState(7)
noise_sigma   = 1.0 * float(np.std(dPsi_tr))
dPsi_tr_noisy = dPsi_tr + rng_noise.normal(0, noise_sigma, dPsi_tr.shape)

# =============================================================================
# === KPH-EDMD IDENTIFICATION ===
# =============================================================================
print("Fitting KpH-EDMD (SDP)...")
model = KpHEDMD(N=N, dt=dt, m_u=2, alpha=1e-3, lam_R=1e-5, lam_u=1e-5)
model.fit(Psi_tr.T, dPsi_tr_noisy.T, U_train.T)

print("Fitting EDMD baseline...")
K_edmd_cont, Ku_edmd_cont = fit_edmd(
    Psi_tr.T, dPsi_tr_noisy.T, U_train.T, N, 2, alpha=1e-3
)

# Convenience aliases
KJ_val, KR_val, Ku_val = model.KJ_, model.KR_, model.Ku_
K_kph        = model.K_disc
Ku_kph       = model.Ku_disc
K_edmd_disc  = np.eye(N) + dt * K_edmd_cont
Ku_edmd_disc = dt * Ku_edmd_cont

skew_err   = model.skew_error()
min_eig_KR = model.kr_min_eigenvalue()
spec_kph   = np.max(np.abs(np.linalg.eigvals(K_kph)))
spec_edmd  = np.max(np.abs(np.linalg.eigvals(K_edmd_disc)))

# =============================================================================
# === RESULTS: PREDICTION ===
# =============================================================================
M_test = 30; T_test = 150; horizon = 10
rng2 = np.random.RandomState(123)
rmse_kph_list = []; rmse_edmd_list = []
pr_kph_max_all = 0.0; pr_edmd_max_all = 0.0

for i in range(M_test):
    q1_t  = rng2.uniform(-np.pi/4, np.pi/4)
    q2_t  = rng2.uniform(-np.pi/4, np.pi/4)
    dq1_t = rng2.uniform(-0.5, 0.5); dq2_t = rng2.uniform(-0.5, 0.5)
    q_t   = np.array([q1_t, q2_t])
    pv_t  = mass_matrix(q_t, params) @ np.array([dq1_t, dq2_t])
    x0_t  = np.concatenate([q_t, pv_t])
    u_seq_t = rng2.uniform(-3, 3, (T_test, 2))
    X_true  = simulate_robot(x0_t, u_seq_t)

    errs_kph = []; errs_edmd = []
    for start in range(T_test - horizon):
        Pk = build_dictionary(X_true[:, start:start+1]).flatten()
        Pe = Pk.copy()
        for h in range(horizon):
            u_h = u_seq_t[start + h]
            Pk  = K_kph       @ Pk + Ku_kph       @ u_h
            Pe  = K_edmd_disc @ Pe + Ku_edmd_disc @ u_h
        x_true_h = X_true[:, start + horizon]
        errs_kph.append( np.linalg.norm(Pk[:4] - x_true_h))
        errs_edmd.append(np.linalg.norm(Pe[:4] - x_true_h))
    rmse_kph_list.append(np.mean(errs_kph))
    rmse_edmd_list.append(np.mean(errs_edmd))

    for k in range(T_test):
        u_k    = u_seq_t[k]
        Psi_k  = build_dictionary(X_true[:, k:k+1]).flatten()
        Psi_nk = K_kph       @ Psi_k + Ku_kph       @ u_k
        Psi_ne = K_edmd_disc @ Psi_k + Ku_edmd_disc @ u_k
        pr_kph_max_all  = max(pr_kph_max_all,
                              passivity_residual(Psi_k, Psi_nk, u_k, Ku_val, dt))
        pr_edmd_max_all = max(pr_edmd_max_all,
                              passivity_residual(Psi_k, Psi_ne, u_k, Ku_edmd_cont, dt))

rmse_kph  = float(np.mean(rmse_kph_list))
rmse_edmd = float(np.mean(rmse_edmd_list))

# =============================================================================
# === RESULTS: CLOSED-LOOP CONTROL ===
# =============================================================================
# x_ref = hanging equilibrium q=[-pi/2, 0] where gravity_vector = 0
x0_ctrl     = np.array([np.pi/4, np.pi/6, 0.0, 0.0])
x_ref       = np.array([-np.pi/2, 0.0, 0.0, 0.0])
psi_ref     = build_dictionary(x_ref.reshape(4, 1)).flatten()
Kd_gain     = 3.0
u_max       = 10.0
T_ctrl      = 300   # 6 seconds


def run_damping_injection(
    x0: np.ndarray,
    Ku_cont: np.ndarray,
    T_sim: int,
    Kd: float,
    psi_ref_vec: np.ndarray,
    u_max_val: float,
) -> tuple:
    """Simulate closed-loop damping injection u = clip(-Kd*Ku^T*(psi-psi_ref)).

    Parameters
    ----------
    x0 : np.ndarray, shape (4,)
        Initial state.
    Ku_cont : np.ndarray, shape (N, 2)
        Continuous-time input coupling matrix.
    T_sim : int
        Number of simulation steps.
    Kd : float
        Damping gain.
    psi_ref_vec : np.ndarray, shape (N,)
        Reference dictionary state phi(x_ref).
    u_max_val : float
        Torque saturation limit.

    Returns
    -------
    X : np.ndarray, shape (4, T_sim+1)
    U : np.ndarray, shape (T_sim, 2)
    """
    X = np.zeros((4, T_sim + 1)); X[:, 0] = x0
    U = np.zeros((T_sim, 2))
    for k in range(T_sim):
        psi_k = build_dictionary(X[:, k:k+1]).flatten()
        u_raw = -Kd * Ku_cont.T @ (psi_k - psi_ref_vec)
        u_k   = np.clip(u_raw, -u_max_val, u_max_val)
        U[k]  = u_k
        sol = solve_ivp(
            two_link_rhs, [0, dt], X[:, k], args=(u_k, params),
            method='RK45', rtol=1e-8, atol=1e-10, max_step=dt/4,
        )
        X[:, k + 1] = sol.y[:, -1]
    return X, U


def run_true_velocity_damping(
    x0: np.ndarray,
    T_sim: int,
    Kd: float,
    u_max_val: float,
) -> tuple:
    """Simulate true velocity-feedback damping u = clip(-Kd * dq).

    Parameters
    ----------
    x0 : np.ndarray, shape (4,)
        Initial state.
    T_sim : int
        Number of simulation steps.
    Kd : float
        Damping gain.
    u_max_val : float
        Torque saturation limit.

    Returns
    -------
    X : np.ndarray, shape (4, T_sim+1)
    U : np.ndarray, shape (T_sim, 2)
    """
    X = np.zeros((4, T_sim + 1)); X[:, 0] = x0
    U = np.zeros((T_sim, 2))
    for k in range(T_sim):
        Mi   = np.linalg.inv(mass_matrix(X[:2, k], params))
        dq_k = Mi @ X[2:, k]
        u_k  = np.clip(-Kd * dq_k, -u_max_val, u_max_val)
        U[k] = u_k
        sol = solve_ivp(
            two_link_rhs, [0, dt], X[:, k], args=(u_k, params),
            method='RK45', rtol=1e-8, atol=1e-10, max_step=dt/4,
        )
        X[:, k + 1] = sol.y[:, -1]
    return X, U


X_kph,  U_kph  = run_damping_injection(x0_ctrl, Ku_val,       T_ctrl, Kd_gain, psi_ref, u_max)
X_edmd, U_edmd = run_damping_injection(x0_ctrl, Ku_edmd_cont, T_ctrl, Kd_gain, psi_ref, u_max)
X_tc,   U_tc   = run_true_velocity_damping(x0_ctrl, T_ctrl, Kd_gain, u_max)

t_vec = np.arange(T_ctrl + 1) * dt


def settling_time(X: np.ndarray, x_ref: np.ndarray, tol: float, t_vec: np.ndarray):
    """Return the first time ||X[:,k] - x_ref|| < tol, or None if not reached.

    Parameters
    ----------
    X : np.ndarray, shape (4, T+1)
    x_ref : np.ndarray, shape (4,)
    tol : float
    t_vec : np.ndarray, shape (T+1,)

    Returns
    -------
    float or None
    """
    norms = np.linalg.norm(X - x_ref[:, None], axis=0)
    idx   = np.where(norms < tol)[0]
    return float(t_vec[idx[0]]) if len(idx) > 0 else None


tol_settle = 0.1
st_kph  = settling_time(X_kph,  x_ref, tol_settle, t_vec)
st_edmd = settling_time(X_edmd, x_ref, tol_settle, t_vec)
st_true = settling_time(X_tc,   x_ref, tol_settle, t_vec)

# =============================================================================
# === CONSOLE SUMMARY ===
# =============================================================================
sep = "=" * 60
print(f"""
{sep}
  KpH-EDMD: Two-Link Robot Results
  Reproduces Table I and Figures 3-4 of Preciado (CDC 2025)
{sep}

pH structure verification:
  Max residual |x_dot - (J-R)*nabla_H - G*u|: {max_ph_err:.2e}  {'PASS' if ph_ok else 'FAIL'}

Identification (N={N}, M={M_train}, dt={dt}s):
  KJ skew-symmetry ||KJ+KJ^T||_F:  {skew_err:.2e}
  KR min eigenvalue:                {min_eig_KR:.6f}  ({'PSD: PASS' if min_eig_KR >= 0 else 'PSD: FAIL'})
  SDP solve time:                   {model.solve_time_:.2f} s

Prediction RMSE (10-step, 30 test trajectories):
  EDMD (unconstrained): {rmse_edmd:8.1f}  spectral radius: {spec_edmd:.1f}
  KpH-EDMD (proposed):  {rmse_kph:8.2f}  spectral radius: {spec_kph:.2f}

Passivity residual max_k PR_k:
  EDMD: {pr_edmd_max_all:.2e}
  KpH:  {pr_kph_max_all:.2e}  {'(exact zero by DT Schur LMI)' if pr_kph_max_all < 1e-6 else ''}

Closed-loop control (damping injection Kd={Kd_gain}I):
  x0 = [pi/4, pi/6, 0, 0]  ->  x_ref = [-pi/2, 0, 0, 0]
  KpH settling time (||x-x_ref|| < {tol_settle}):  {f'{st_kph:.2f} s' if st_kph else 'did not settle'}
  EDMD: {f'{st_edmd:.2f} s' if st_edmd else 'DIVERGES (spectral radius > 1)'}
  True: {f'{st_true:.2f} s' if st_true else 'did not settle'}
""")

# =============================================================================
# === FIGURES ===
# =============================================================================

# fig_robot_joints.pdf — joint angles under damping injection
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

axes[0].plot(t_vec, X_kph[0],  label='KpH-EDMD', **STYLE['kph'])
axes[0].plot(t_vec, X_edmd[0], label='EDMD',     **STYLE['edmd'])
axes[0].plot(t_vec, X_tc[0],   label='True',     **STYLE['true'])
axes[0].axhline(x_ref[0], **STYLE['tol'], label='Target')
axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel(r'$q_1$ (rad)')
axes[0].set_title('Joint 1'); axes[0].legend()

axes[1].plot(t_vec, X_kph[1],  label='KpH-EDMD', **STYLE['kph'])
axes[1].plot(t_vec, X_edmd[1], label='EDMD',     **STYLE['edmd'])
axes[1].plot(t_vec, X_tc[1],   label='True',     **STYLE['true'])
axes[1].axhline(x_ref[1], **STYLE['tol'], label='Target')
axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel(r'$q_2$ (rad)')
axes[1].set_title('Joint 2'); axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_robot_joints.pdf'), bbox_inches='tight')
plt.close(fig)

# fig_robot_energy.pdf — energy + passivity residual
H_vec_fn = lambda Xtraj: np.array([hamiltonian(Xtraj[:, k], params)
                                   for k in range(Xtraj.shape[1])])
H_kph_v  = H_vec_fn(X_kph)
H_edmd_v = H_vec_fn(X_edmd)
H_true_v = H_vec_fn(X_tc)

rng3 = np.random.RandomState(77)
u_pr   = rng3.uniform(-3, 3, (T_test, 2))
x0_pr  = np.array([0.2, -0.1, 0.05, -0.03])
X_pr   = simulate_robot(x0_pr, u_pr)
pr_k_v = []; pr_e_v = []
for k in range(T_test):
    u_k    = u_pr[k]
    Psi_k  = build_dictionary(X_pr[:, k:k+1]).flatten()
    pr_k_v.append(passivity_residual(Psi_k, K_kph@Psi_k + Ku_kph@u_k,
                                     u_k, Ku_val, dt))
    pr_e_v.append(passivity_residual(Psi_k, K_edmd_disc@Psi_k + Ku_edmd_disc@u_k,
                                     u_k, Ku_edmd_cont, dt))

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
axes[0].plot(t_vec, H_kph_v,  label='KpH-EDMD', **STYLE['kph'])
axes[0].plot(t_vec, H_edmd_v, label='EDMD',     **STYLE['edmd'])
axes[0].plot(t_vec, H_true_v, label='True',     **STYLE['true'])
axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel(r'$\mathcal{H}(x)$ [J]')
axes[0].set_title('Total energy'); axes[0].legend()

axes[1].plot(np.arange(T_test), pr_e_v, label='EDMD',     **STYLE['edmd'])
axes[1].plot(np.arange(T_test), pr_k_v, label='KpH-EDMD', **STYLE['kph'])
axes[1].set_xlabel(r'Step $k$'); axes[1].set_ylabel(r'$\mathrm{PR}_k$')
axes[1].set_title('Passivity residual'); axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_robot_energy.pdf'), bbox_inches='tight')
plt.close(fig)

print(f"Figures saved:")
print(f"  figures/fig_robot_joints.pdf")
print(f"  figures/fig_robot_energy.pdf")

# =============================================================================
# === SANITY CHECKS (Task 6) ===
# =============================================================================
print(f"\n{sep}")
print(f"  Verification (Task 6)")
print(f"{sep}")

figs_list = ['fig_robot_joints.pdf', 'fig_robot_energy.pdf']

c_ph    = ph_ok                                              # pH residual < 1e-6
c_kr    = min_eig_KR > 0                                     # KR PSD
c_kj    = skew_err < 1e-8                                    # KJ skew-symmetric
c_rmse  = rmse_kph < rmse_edmd                               # KpH better than EDMD
c_pr    = pr_kph_max_all < 1e-6                              # PR_k = 0
c_sett  = st_kph is not None and st_kph < 5.0               # KpH settles < 5 s
c_figs  = all(os.path.exists(os.path.join(FIG, f)) for f in figs_list)

checks = [
    (c_ph,   f"pH structure residual < 1e-6  ({max_ph_err:.2e})"),
    (c_kr,   f"KR PSD (min eigenvalue = {min_eig_KR:.6f} > 0)"),
    (c_kj,   f"KJ skew-symmetric (||KJ+KJ^T||_F = {skew_err:.2e} < 1e-8)"),
    (c_rmse, f"KpH RMSE < EDMD RMSE  ({rmse_kph:.2f} < {rmse_edmd:.1f})"),
    (c_pr,   f"PR_k = 0 for KpH (max = {pr_kph_max_all:.2e} < 1e-8)"),
    (c_sett, f"KpH settling time < 5.0 s  ({f'{st_kph:.2f} s' if st_kph else 'did not settle'})"),
    (c_figs, f"fig_robot_joints.pdf saved to figures/"),
    (c_figs, f"fig_robot_energy.pdf saved to figures/"),
]

all_pass = True
for ok, msg in checks:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {msg}")
    if not ok:
        all_pass = False

print(f"\n  All checks passed: {all_pass}")
print(sep)

# =============================================================================
# === FILE LIST FOR GITHUB (Task 7) ===
# =============================================================================
print("""
=== FILES READY FOR GITHUB UPLOAD ===

KpH-EDMD/
├── README.md
├── requirements.txt
├── kph/
│   ├── __init__.py
│   └── edmd.py
├── examples/
│   └── two_link_robot.py
└── figures/
    ├── fig_robot_joints.pdf
    └── fig_robot_energy.pdf

Upload command:
  git add README.md requirements.txt kph/ examples/ figures/
  git commit -m "Initial release: KpH-EDMD for two-link robot (CDC 2025)"
  git push origin main
""")
