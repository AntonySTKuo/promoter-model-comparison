"""
Model.py  —  Thermodynamic models for TF-regulated gene expression.

Parameter names follow Model_Architectures.png:

    r_max   : maximum expression rate (RNAP-bound state)
    r0      : basal expression rate (empty-promoter state)  [default = 0]
    P_T     : TF occupancy  =  [TF] / K_T
    beta    : cooperativity coefficient β (TF–RNAP ternary complex weight)
    alpha   : ternary-complex expression coefficient α  (GM only)
    P_P     : RNAP occupancy  =  [RNAP] / K_P  (per-datapoint latent variable)

All scalar parameters are stored as log₁₀ values in nn.Parameter.
P_P is stored as a per-datapoint log₁₀ latent: P_latent.

Note: 'beta' here is the cooperativity β from the figure,
      NOT the thermodynamic β = 1/(kB·T).
      This model works directly with occupancy probabilities,
      so thermodynamic β does not appear.

Model variants:
    Model_GM  —  Generalized Model  (4-state, α trainable or fixed)
    Model_RM  —  Repression Model   (3-state, no ternary complex)
    Model_AM  —  Activation Model   (4-state, α = 1 fixed)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist


# Initial value of P_T (linear scale) shared by all three models. Datasets that
# need a different starting point override it per fit via init_PT.
DEFAULT_INIT_PT = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Shared base
# ─────────────────────────────────────────────────────────────────────────────

class _BaseModel(nn.Module):
    """
    Shared infrastructure for all three thermodynamic models.

    Parameters defined here:
        log_rmax  : log10(r_max)
        log_r0    : log10(r0)    [only created when include_r0=True; else r0 = 0]
        P_latent  : log10(P_P) per datapoint  (one latent value per observation)
    """

    def __init__(self, num_P_points, include_r0=False):
        super().__init__()
        self.include_r0 = include_r0

        # Per-datapoint latent RNAP occupancy (log10 space, uniform init in [-3, 3])
        self.P_latent = nn.Parameter(torch.rand(num_P_points) * 6 - 3)

        # Global expression scale
        self.log_rmax = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))

        if include_r0:
            self.log_r0 = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

    # ── helper: return r₀ in linear scale ────────────────────────────────────

    def _r0(self):
        """r₀ in linear scale. Returns Python float 0.0 if include_r0=False."""
        return torch.pow(10.0, self.log_r0) if self.include_r0 else 0.0

    # ── constitutive expression (no TF present) ───────────────────────────────

    def con_model(self, P):
        """
        E_TF⁻  =  (r0 + r_max · P_P) / (1 + P_P)    [returned as log10]
        """
        r_max = torch.pow(10.0, self.log_rmax)
        r0    = self._r0()
        return torch.log10((r0 + r_max * P) / (1.0 + P))

    # ── regulated expression (TF present) ────────────────────────────────────
    # Subclasses must implement reg_model(P).

    def reg_model(self, P):
        raise NotImplementedError

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self):
        P        = torch.pow(10.0, self.P_latent)
        con_pred = self.con_model(P)
        reg_pred = self.reg_model(P)
        return con_pred, reg_pred, P


# ─────────────────────────────────────────────────────────────────────────────
# Model_GM  —  Generalized Model (4-state)
# ─────────────────────────────────────────────────────────────────────────────

class Model_GM(_BaseModel):
    """
    Generalized Model (GM)  —  4-state.

    State table:
        State             Weight          Expression
        ─────────────── ──────────────── ────────────
        empty             1               r0
        RNAP bound        P_P             r_max
        TF bound          P_T             r0
        RNAP + TF (ternary) β·P_P·P_T    α·r_max

    Regulated expression:
        E_TF⁺ = [r0·(1 + P_T) + r_max·(P_P + α·β·P_P·P_T)]
                ────────────────────────────────────────────
                        1 + P_P + P_T + β·P_P·P_T

    Parameters (all log10):
        log_rmax  : log10(r_max)
        log_r0    : log10(r0)    [only if include_r0=True]
        log_PT    : log10(P_T)
        log_beta  : log10(β)     cooperativity
        log_alpha : log10(α)     ternary-complex expression coefficient
                    • alpha_fixed=False (default) → log_alpha is trained
                    • alpha_fixed=True            → log_alpha frozen at log10(alpha_value);
                                                    parameter still appears in state_dict
                                                    so checkpoints stay consistent
    """

    def __init__(self, num_P_points, include_r0=False, alpha_fixed=False,
                 init_beta=1.0, init_alpha=1.0, init_PT=DEFAULT_INIT_PT):
        super().__init__(num_P_points, include_r0)
        self.alpha_fixed = alpha_fixed

        self.log_PT    = nn.Parameter(torch.tensor(np.log10(init_PT), dtype=torch.float32))
        self.log_beta  = nn.Parameter(torch.tensor(np.log10(init_beta),  dtype=torch.float32))
        self.log_alpha = nn.Parameter(torch.tensor(np.log10(init_alpha), dtype=torch.float32),
                                      requires_grad=not alpha_fixed)

    def _alpha(self):
        """α in linear scale."""
        return torch.pow(10.0, self.log_alpha)

    def reg_model(self, P):
        r_max = torch.pow(10.0, self.log_rmax)
        r0    = self._r0()
        P_T   = torch.pow(10.0, self.log_PT)
        beta  = torch.pow(10.0, self.log_beta)
        alpha = self._alpha()

        numerator   = r0 * (1.0 + P_T) + r_max * (P + alpha * beta * P * P_T)
        denominator = 1.0 + P + P_T + beta * P * P_T
        return torch.log10(numerator / denominator)


# ─────────────────────────────────────────────────────────────────────────────
# Model_RM  —  Repression Model (3-state)
# ─────────────────────────────────────────────────────────────────────────────

class Model_RM(_BaseModel):
    """
    Repression Model (RM)  —  3-state.

    State table:
        State        Weight   Expression
        ──────────── ──────── ──────────
        empty         1        r0
        RNAP bound    P_P      r_max
        TF bound      P_T      r0         (TF sterically excludes RNAP; no ternary complex)

    Regulated expression:
        E_TF⁺ = [r0·(1 + P_T) + r_max·P_P]
                ─────────────────────────────
                       1 + P_P + P_T

    Parameters (all log10):
        log_rmax : log10(r_max)
        log_r0   : log10(r0)   [only if include_r0=True]
        log_PT   : log10(P_T)
    """

    def __init__(self, num_P_points, include_r0=True, init_PT=DEFAULT_INIT_PT):
        super().__init__(num_P_points, include_r0)
        self.log_PT = nn.Parameter(torch.tensor(np.log10(init_PT), dtype=torch.float32))

    def reg_model(self, P):
        r_max = torch.pow(10.0, self.log_rmax)
        r0    = self._r0()
        P_T   = torch.pow(10.0, self.log_PT)

        numerator   = r0 * (1.0 + P_T) + r_max * P
        denominator = 1.0 + P + P_T
        return torch.log10(numerator / denominator)


# ─────────────────────────────────────────────────────────────────────────────
# Model_AM  —  Activation Model (4-state, α = 1 fixed)
# ─────────────────────────────────────────────────────────────────────────────

class Model_AM(_BaseModel):
    """
    Activation Model (AM)  —  4-state, α = 1 (fixed).

    State table:
        State              Weight         Expression
        ────────────────── ─────────────── ──────────
        empty               1               r0
        RNAP bound          P_P             r_max
        TF bound            P_T             r0
        RNAP + TF (ternary) β·P_P·P_T      r_max      (α = 1)

    Regulated expression:
        E_TF⁺ = [r0·(1 + P_T) + r_max·(P_P + β·P_P·P_T)]
                ───────────────────────────────────────────
                        1 + P_P + P_T + β·P_P·P_T

    Parameters (all log10):
        log_rmax : log10(r_max)
        log_r0   : log10(r0)   [only if include_r0=True]
        log_PT   : log10(P_T)
        log_beta : log10(β)    cooperativity
    """

    def __init__(self, num_P_points, include_r0=True, init_PT=DEFAULT_INIT_PT):
        super().__init__(num_P_points, include_r0)
        self.log_PT   = nn.Parameter(torch.tensor(np.log10(init_PT), dtype=torch.float32))
        self.log_beta = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))

    def reg_model(self, P):
        """α = 1 fixed: ternary complex has same expression level as RNAP-only state."""
        r_max = torch.pow(10.0, self.log_rmax)
        r0    = self._r0()
        P_T   = torch.pow(10.0, self.log_PT)
        beta  = torch.pow(10.0, self.log_beta)

        numerator   = r0 * (1.0 + P_T) + r_max * (P + beta * P * P_T)  # α = 1
        denominator = 1.0 + P + P_T + beta * P * P_T
        return torch.log10(numerator / denominator)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry  (used for cache deserialization)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CLASSES = {
    'Model_GM': Model_GM,
    'Model_RM': Model_RM,
    'Model_AM': Model_AM,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────

def modelFit(df, model_arch, num_epochs=5000, lr=0.005, latent_l2=0.0, **model_kwargs):
    """
    Fit a thermodynamic model to paired (ConExp, RegExp) observations.

    Args:
        df          : DataFrame with columns 'ConExp' and 'RegExp' (linear scale).
        model_arch  : Model_GM, Model_RM, or Model_AM.
        num_epochs  : Training epochs (default 5000).
        lr          : Adam learning rate (default 0.005).
        latent_l2   : Weight of a variance penalty on P_latent, lambda * mean((p - p_bar)^2).
                      0.0 (default) leaves the loss untouched.

                      Where the fitted curve has saturated, both con_model and
                      reg_model are flat in P_P, so those points' latents carry
                      no gradient and drift to arbitrary values (P_P up to 1e5
                      for the CpxR datasets, where the curve moves only 0.04 dex
                      between P_P = 1e1 and 1e5). The penalty resolves that
                      non-identifiability: a latent settles where the data
                      gradient balances it, so points on a steep part of the
                      curve barely move while flat-region points are pulled in.
                      Centring on the dataset's own mean rather than on zero
                      makes the penalty translation-invariant, so it constrains
                      only how far the latents spread, not where they sit.
        **model_kwargs : Forwarded to model_arch.__init__  (e.g. alpha_fixed=True).

    Returns:
        Fitted model in eval() mode.
    """
    con_obs = torch.tensor(np.log10(df['ConExp'].values), dtype=torch.float32)
    reg_obs = torch.tensor(np.log10(df['RegExp'].values), dtype=torch.float32)

    model     = model_arch(num_P_points=len(con_obs), **model_kwargs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"  Fitting {model_arch.__name__}  "
          f"({len(con_obs)} pts, {num_epochs} epochs) ...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        con_pred, reg_pred, _ = model()
        loss = criterion(con_pred, con_obs) + criterion(reg_pred, reg_obs)
        if latent_l2:
            loss = loss + latent_l2 * torch.var(model.P_latent, unbiased=False)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"    epoch {epoch:5d}/{num_epochs}  loss = {loss.item():.6f}")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(model, TF, X_obs, Y_obs, p_range=(-10, 10), res=5000):
    """
    Compute five metrics, all in log10 scale, simultaneously over (ConExp, RegExp).

    Orthogonal-distance metrics  (curve-based):
        geo_RMSE  — geometric RMSE: sqrt of mean squared orthogonal distance
                    from each 2D observation to the nearest point on the curve
        TLS       — total least squares: sum of squared orthogonal distances

    Direct-prediction metrics  (P_latent-based, all N×2 points pooled):
        R²        — coefficient of determination
        Pearson_r — Pearson correlation coefficient
        MAE       — mean absolute error

    Args:
        X_obs : log10(ConExp) array  (N,)
        Y_obs : log10(RegExp) array  (N,)

    Returns:
        dict with keys: geo_rmse, tls, r2, pearson_r, mae
    """
    # ── Orthogonal-distance metrics ──────────────────────────────────────────
    Ps = torch.tensor(np.logspace(p_range[0], p_range[1], res), dtype=torch.float32)
    with torch.no_grad():
        X_curve = model.con_model(Ps).numpy()
        Y_curve = model.reg_model(Ps).numpy()

    valid = np.isfinite(X_curve) & np.isfinite(Y_curve)
    pts_curve = np.column_stack((X_curve[valid], Y_curve[valid]))
    pts_obs   = np.column_stack((X_obs, Y_obs))

    min_dists = cdist(pts_obs, pts_curve, metric='euclidean').min(axis=1)
    tls      = float(np.sum(min_dists ** 2))
    geo_rmse = float(np.sqrt(np.mean(min_dists ** 2)))

    # ── Direct-prediction metrics (using fitted P_latent) ────────────────────
    P = torch.pow(10.0, model.P_latent.detach())
    with torch.no_grad():
        con_pred = model.con_model(P).numpy()
        reg_pred = model.reg_model(P).numpy()

    # Pool all N×2 predictions and observations
    pred = np.concatenate([con_pred, reg_pred])
    obs  = np.concatenate([X_obs,    Y_obs])

    ss_res    = np.sum((pred - obs) ** 2)
    ss_tot    = np.sum((obs  - obs.mean()) ** 2)
    r2        = float(1.0 - ss_res / ss_tot)
    pearson_r = float(np.corrcoef(pred, obs)[0, 1])
    mae       = float(np.mean(np.abs(pred - obs)))

    print(f"  Metrics [{type(model).__name__}]:  "
          f"geo_rmse={geo_rmse:.4f}  tls={tls:.4f}  "
          f"R2={r2:.4f}  r={pearson_r:.4f}  mae={mae:.4f}")

    return {'geo_rmse': geo_rmse, 'tls': tls, 'r2': r2, 'pearson_r': pearson_r, 'mae': mae}
