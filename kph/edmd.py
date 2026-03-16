"""
kph.edmd — Koopman-Port-Hamiltonian Extended Dynamic Mode Decomposition
=======================================================================

Solves the structured system identification problem (eq. 10 in Preciado 2025):

    min_{KJ, KR, Ku}  sum_k ||dPsi_k/dt - (KJ - KR)*Psi_k - Ku*u_k||^2
                      + alpha*||KJ - KR||_F^2
                      + lam_R*||KR||_1
                      + lam_u*||Ku||_1

    subject to:
        KJ + KJ^T = 0                                (skew-symmetry, J structure)
        KR = KR^T >= 0                               (PSD, R structure)
        [[I, A_d^T], [A_d, I]] >> schur_eps * I     (DT Schur LMI: passivity)
        where A_d = I + dt*(KJ - KR)

Passivity guarantee (Theorem 1 of the paper):
    The DT Schur LMI A_d^T A_d < I guarantees PR_k = 0 at every step:

        (H_{k+1} - H_k) / dt  -  Psi_{k+1}^T Ku u_k  <=  0

    where H_k = 0.5 * ||Psi_k||^2 is the lifted-space Hamiltonian proxy.

References
----------
V. M. Preciado, "A Koopman-Port-Hamiltonian Framework for Data-Driven
Modeling and Control," IEEE Conference on Decision and Control, 2025.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import cvxpy as cp

try:
    import mosek          # noqa: F401
    SOLVER = cp.MOSEK
    MOSEK_AVAILABLE = True
except ImportError:
    SOLVER = cp.CLARABEL
    MOSEK_AVAILABLE = False


class KpHEDMD:
    """Koopman-Port-Hamiltonian Extended Dynamic Mode Decomposition.

    Identifies a discrete-time Koopman approximation with enforced
    port-Hamiltonian structure (skew-symmetric J + PSD R) via a
    convex SDP with a passivity-guaranteeing Schur LMI.

    The continuous-time Koopman generator is decomposed as::

        F = KJ - KR

    where KJ is skew-symmetric (conservative J structure) and KR >= 0
    (dissipative R structure). The discrete-time model is::

        Psi_{k+1} = (I + dt*F) Psi_k + dt*Ku u_k

    Passivity is enforced via the DT Schur LMI::

        [[I, A_d^T], [A_d, I]] >> schur_eps*I   <==>   A_d^T A_d < I

    Parameters
    ----------
    N : int
        Dictionary (feature) dimension.
    dt : float
        Sampling period in seconds.
    m_u : int, optional
        Number of control inputs. Default 1.
    alpha : float, optional
        Tikhonov regularization on ||F||_F^2. Default 1e-3.
    lam_R : float, optional
        L1 sparsity weight on KR. Default 1e-5.
    lam_u : float, optional
        L1 sparsity weight on Ku. Default 1e-5.
    schur_eps : float, optional
        Strict feasibility margin for the DT Schur LMI. Default 1e-6.

    Attributes
    ----------
    KJ_ : np.ndarray, shape (N, N)
        Fitted skew-symmetric Koopman J matrix (continuous time).
    KR_ : np.ndarray, shape (N, N)
        Fitted PSD Koopman R matrix (continuous time).
    Ku_ : np.ndarray, shape (N, m_u)
        Fitted input coupling matrix (continuous time).
    solve_time_ : float
        Wall-clock time for the SDP solve in seconds.
    """

    def __init__(
        self,
        N: int,
        dt: float,
        m_u: int = 1,
        alpha: float = 1e-3,
        lam_R: float = 1e-5,
        lam_u: float = 1e-5,
        schur_eps: float = 1e-6,
    ) -> None:
        self.N = N
        self.dt = dt
        self.m_u = m_u
        self.alpha = alpha
        self.lam_R = lam_R
        self.lam_u = lam_u
        self.schur_eps = schur_eps

        self.KJ_: Optional[np.ndarray] = None
        self.KR_: Optional[np.ndarray] = None
        self.Ku_: Optional[np.ndarray] = None
        self.solve_time_: float = 0.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        Psi: np.ndarray,
        dPsi_dt: np.ndarray,
        U: np.ndarray,
    ) -> "KpHEDMD":
        """Solve the KpH-EDMD SDP and store KJ_, KR_, Ku_.

        Parameters
        ----------
        Psi : np.ndarray, shape (M, N)
            Dictionary snapshots at current time (row-major: M samples).
        dPsi_dt : np.ndarray, shape (M, N)
            Time derivatives of the dictionary snapshots.
        U : np.ndarray, shape (M, m_u)
            Control inputs corresponding to each snapshot.

        Returns
        -------
        self : KpHEDMD
            Fitted estimator (supports method chaining).

        Notes
        -----
        Uses MOSEK when available; falls back to CLARABEL automatically.
        KJ_ is post-processed to be exactly skew-symmetric::

            KJ_ = (KJ_raw - KJ_raw^T) / 2
        """
        N, dt = self.N, self.dt

        KJ = cp.Variable((N, N))
        KR = cp.Variable((N, N), symmetric=True)
        Ku = cp.Variable((N, self.m_u))
        F  = cp.Variable((N, N))

        res = dPsi_dt.T - F @ Psi.T - Ku @ U.T
        obj = (
            cp.sum_squares(res)
            + self.alpha * cp.sum_squares(F)
            + self.lam_R * cp.norm1(KR)
            + self.lam_u * cp.norm1(Ku)
        )

        A_d_expr = np.eye(N) + dt * F
        schur = cp.bmat([
            [np.eye(N),  A_d_expr.T],
            [A_d_expr,   np.eye(N)],
        ])
        constraints = [
            KJ + KJ.T == 0,
            KR - KR.T == 0,
            KR >> 0,
            F == KJ - KR,
            schur >> self.schur_eps * np.eye(2 * N),
        ]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        t0 = time.time()
        try:
            prob.solve(solver=SOLVER, verbose=False)
        except Exception:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        self.solve_time_ = time.time() - t0

        self.KJ_ = (KJ.value - KJ.value.T) / 2
        self.KR_ = KR.value.copy()
        self.Ku_ = Ku.value.copy()
        return self

    # ------------------------------------------------------------------
    # Derived discrete-time matrices
    # ------------------------------------------------------------------

    @property
    def K_disc(self) -> np.ndarray:
        """Discrete-time state transition matrix A_d = I + dt*(KJ - KR).

        Returns
        -------
        np.ndarray, shape (N, N)
        """
        self._check_fitted()
        return np.eye(self.N) + self.dt * (self.KJ_ - self.KR_)

    @property
    def Ku_disc(self) -> np.ndarray:
        """Discrete-time input coupling matrix dt*Ku.

        Returns
        -------
        np.ndarray, shape (N, m_u)
        """
        self._check_fitted()
        return self.dt * self.Ku_

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        psi0: np.ndarray,
        U_seq: np.ndarray,
    ) -> np.ndarray:
        """Roll out the discrete-time KpH model from an initial lifted state.

        Parameters
        ----------
        psi0 : np.ndarray, shape (N,)
            Initial dictionary state Psi_0 = phi(x_0).
        U_seq : np.ndarray, shape (T, m_u)
            Control inputs for T steps.

        Returns
        -------
        Psi_traj : np.ndarray, shape (N, T+1)
            Predicted dictionary trajectory; columns correspond to time steps.
        """
        self._check_fitted()
        T = U_seq.shape[0]
        Psi = np.zeros((self.N, T + 1))
        Psi[:, 0] = psi0.flatten()
        A, B = self.K_disc, self.Ku_disc
        for k in range(T):
            Psi[:, k + 1] = A @ Psi[:, k] + B @ U_seq[k]
        return Psi

    # ------------------------------------------------------------------
    # Structure diagnostics
    # ------------------------------------------------------------------

    def skew_error(self) -> float:
        """Return ||KJ + KJ^T||_F; should be ~0 by construction.

        Returns
        -------
        float
        """
        self._check_fitted()
        return float(np.linalg.norm(self.KJ_ + self.KJ_.T, "fro"))

    def kr_min_eigenvalue(self) -> float:
        """Return the minimum eigenvalue of KR; must be >= 0 for PSD.

        Returns
        -------
        float
        """
        self._check_fitted()
        return float(np.linalg.eigvalsh(self.KR_).min())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.KJ_ is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() before accessing model attributes."
            )


# ---------------------------------------------------------------------------
# Standalone utilities
# ---------------------------------------------------------------------------

def fit_edmd(
    Psi: np.ndarray,
    dPsi_dt: np.ndarray,
    U: np.ndarray,
    N: int,
    m_u: int,
    alpha: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit an unconstrained EDMD baseline via ridge regression.

    Solves the normal equations::

        [K_cont, Ku_cont] = dPsi_dt^T @ [Psi; U] @ inv([Psi; U]^T @ [Psi; U] + alpha*I)

    Parameters
    ----------
    Psi : np.ndarray, shape (M, N)
        Dictionary snapshots (row-major).
    dPsi_dt : np.ndarray, shape (M, N)
        Time derivatives of dictionary snapshots.
    U : np.ndarray, shape (M, m_u)
        Control inputs.
    N : int
        Dictionary dimension.
    m_u : int
        Control input dimension.
    alpha : float, optional
        Ridge regularization weight. Default 1e-3.

    Returns
    -------
    K_cont : np.ndarray, shape (N, N)
        Continuous-time Koopman state matrix (unconstrained).
    Ku_cont : np.ndarray, shape (N, m_u)
        Continuous-time input coupling matrix (unconstrained).
    """
    A = np.hstack([Psi, U])
    sol = np.linalg.solve(A.T @ A + alpha * np.eye(N + m_u), A.T @ dPsi_dt)
    return sol[:N, :].T, sol[N:, :].T


def passivity_residual(
    Psi_k: np.ndarray,
    Psi_next: np.ndarray,
    u_k: np.ndarray,
    Ku_cont: np.ndarray,
    dt: float,
) -> float:
    """Compute the one-step passivity residual PR_k.

    PR_k = max(0,  (H_{k+1} - H_k) / dt  -  Psi_{k+1}^T Ku u_k)

    where H_k = 0.5 * ||Psi_k||^2 is the lifted-space Hamiltonian proxy
    and the supply rate is s_k = Psi_{k+1}^T Ku u_k.

    PR_k = 0 for all k when the DT Schur LMI A_d^T A_d < I is satisfied.

    Parameters
    ----------
    Psi_k : np.ndarray, shape (N,)
        Dictionary state at step k.
    Psi_next : np.ndarray, shape (N,)
        Dictionary state at step k+1 (one-step model prediction).
    u_k : np.ndarray, shape (m_u,)
        Control input at step k.
    Ku_cont : np.ndarray, shape (N, m_u)
        Continuous-time input coupling matrix.
    dt : float
        Sampling period in seconds.

    Returns
    -------
    float
        Non-negative passivity residual (0.0 means passive at this step).
    """
    H_k    = 0.5 * float(np.dot(Psi_k,    Psi_k))
    H_next = 0.5 * float(np.dot(Psi_next, Psi_next))
    supply = float(np.dot(Ku_cont.T @ Psi_next, u_k))
    return max(0.0, (H_next - H_k) / dt - supply)
