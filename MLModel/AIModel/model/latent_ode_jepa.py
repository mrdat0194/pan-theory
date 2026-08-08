"""
latent_ode_jepa.py
==================
Continuous-Time Latent ODE JEPA for Anomaly Detection (Upgrade from MTS-JEPA).

Mathematical Context
--------------------
The original MTS-JEPA used a discrete Transformer (ARPredictor) to predict
the next latent state z_{t+1} from z_t. This assumes equally-spaced, regular
time steps — a significant constraint for real-world sensor data.

This module replaces the Transformer predictor with a **Neural ODE** dynamics
function. The latent state z(t) is now governed by a learned ODE:

    dz(t)/dt = f_theta(z(t))

Given an initial state z(t0), we integrate this ODE forward to any future time
t_k (regular OR irregular) using the Euler / RK4 method without external libraries.

Architecture
------------
    MTS Signal [B, C, T]
         │
    Encoder1D (Conv1D)        ← Same as before
         │
    z_initial [B, D]          ← context vector at t=0 (mean of temporal latents)
         │
    ODEDynamicsNet(z, t)      ← New: f_theta(z, t), a small MLP
         │
    odeint(f, z0, t_span)     ← Numerical integration (Euler or RK4)
         │
    z_predicted [B, D]        ← latent state at target time t_k
         │
    JEPA Loss (MSE + SIGReg)  ← Same as before

Backward Compatibility
-----------------------
The top-level API (build_jepa, compute_anomaly_score, etc.) remains
identical to jepa_backbone.py, so main_jepa_anomaly.py requires
only a single import change.
"""

import torch
import torch.nn as nn
import lejepa


# ─────────────────────────────────────────────────────────────────────────────
# 1. Encoder: Same as original (Conv1D — projects raw signal into latent space)
# ─────────────────────────────────────────────────────────────────────────────
class Encoder1D(nn.Module):
    """
    Maps the raw multi-channel time series [B, C, T] into a latent feature map [B, D, T].
    Identical to the original jepa_backbone.py encoder to maintain compatibility.
    """
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input:  [B, C, T]
        # Output: [B, D, T]
        return self.net(x)


# Expose as stem alias for Le MuMo JEPA compatibility
SequenceStem = Encoder1D


# ─────────────────────────────────────────────────────────────────────────────
# 2. Neural ODE Dynamics: f_theta(z, t) — the derivative of the latent state
# ─────────────────────────────────────────────────────────────────────────────
class ODEDynamicsNet(nn.Module):
    """
    The Neural ODE function f_theta that parameterizes the dynamics of the latent space.

    This small MLP outputs dz/dt = f_theta(z, t). We concatenate [z, t] as input
    so the dynamics can be time-aware (non-autonomous ODE). This allows the
    model to capture behaviours that change over time (e.g., drift, oscillation).

    The key insight: any continuous-time dynamical system that can be written as
    dz/dt = f(z, t) can be learned if we parameterize f with a neural network.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, depth: int = 2):
        super().__init__()
        # The input is the concatenation of [z(t), t], so dim is latent_dim + 1
        layers = [nn.Linear(latent_dim + 1, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute dz/dt at a given time t.
        Args:
            z: latent state [B, D]
            t: scalar time value (float)
        Returns:
            dz_dt: [B, D]
        """
        # Append time as an extra feature column so dynamics are time-aware
        # t_tensor shape: [B, 1]
        t_tensor = torch.full((z.shape[0], 1), t, dtype=z.dtype, device=z.device)
        zt = torch.cat([z, t_tensor], dim=-1)  # [B, D+1]
        return self.net(zt)                     # [B, D]


# ─────────────────────────────────────────────────────────────────────────────
# 3. ODE Solver: Runge-Kutta 4 (RK4) — no external dependency needed
# ─────────────────────────────────────────────────────────────────────────────
def odeint_rk4(f, z0: torch.Tensor, t_start: float, t_end: float, n_steps: int = 10) -> torch.Tensor:
    """
    Integrate the ODE dz/dt = f(z, t) from t_start to t_end using the classical
    4th-order Runge-Kutta (RK4) method.

    Why RK4?
    - It achieves O(h^4) per-step accuracy — much better than the O(h) Euler method.
    - It requires only f-evaluations at 4 points per step, with no Jacobian.
    - It avoids the need for torchdiffeq or other external libraries.

    RK4 Formula (for step from t to t+h):
        k1 = h * f(z,           t)
        k2 = h * f(z + k1/2,   t + h/2)
        k3 = h * f(z + k2/2,   t + h/2)
        k4 = h * f(z + k3,     t + h)
        z_next = z + (k1 + 2*k2 + 2*k3 + k4) / 6

    Args:
        f:       callable(z, t) -> dz/dt
        z0:      initial latent state [B, D]
        t_start: start time (float)
        t_end:   end time (float)
        n_steps: number of integration steps (more steps = more accurate)

    Returns:
        z_end: latent state at t_end, shape [B, D]
    """
    z = z0
    dt = (t_end - t_start) / n_steps

    for i in range(n_steps):
        t = t_start + i * dt

        k1 = f(z,            t)
        k2 = f(z + dt/2 * k1, t + dt/2)
        k3 = f(z + dt/2 * k2, t + dt/2)
        k4 = f(z + dt  * k3,  t + dt)

        z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return z


# ─────────────────────────────────────────────────────────────────────────────
# 4. Latent ODE JEPA: the full model combining Encoder + Neural ODE + SIGReg
# ─────────────────────────────────────────────────────────────────────────────
class LatentODE_JEPA(nn.Module):
    """
    Continuous-Time JEPA for Time-Series Anomaly Detection.

    Upgrade from LeWMJEPA (discrete Transformer):
        BEFORE: ARPredictor (Transformer) steps z[t] -> z[t+1] discretely.
        AFTER:  ODEDynamicsNet integrates dz/dt = f(z, t) continuously.

    The model learns by predicting future latent states at arbitrary future
    times t_k > t_0 and computing the JEPA SSL loss against the actual
    encoded future states.

    This makes it robust to irregular time series:
        - Readings can arrive at t=1.2, t=5.9, t=14.1 (any timestamps)
        - The ODE integrates smoothly to wherever the next reading is
    """
    def __init__(self, encoder: Encoder1D, dynamics: ODEDynamicsNet,
                 sigreg_lambda: float = 1.0, ode_steps: int = 10):
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.sigreg_lambda = sigreg_lambda
        self.ode_steps = ode_steps  # RK4 integration steps per prediction interval

        # LeJEPA SIGReg: prevents representation collapse without EMA or stop-gradient
        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test, num_slices=1024
        )

    def unroll(self, feat: torch.Tensor, actions=None, nsteps: int = 3,
               unroll_mode: str = "parallel", compute_loss: bool = True,
               return_all_steps: bool = False):
        """
        Unroll the Latent ODE JEPA forward, computing predictions and loss.

        Process:
            1. Encode the full signal -> latent feature map [B, D, T]
            2. Take the mean latent at t=0 as the initial ODE state z_0
            3. For each step k in 1..nsteps:
               a. Integrate ODE from t=(k-1) to t=k to predict z_pred(k)
               b. Extract z_target(k) from the encoded feature map
            4. Compute MSE(z_pred, z_target) + SIGReg(z_target)

        This API is identical to LeWMJEPA.unroll() for full compatibility
        with main_jepa_anomaly.py.

        Args:
            feat:          [B, C, T] raw time-series tensor
            actions:       ignored (kept for API compatibility)
            nsteps:        number of future time steps to predict
            unroll_mode:   ignored (kept for API compatibility)
            compute_loss:  if True, return (None, [loss, pred_loss, sigreg_loss])
                           if False, return (predicted_z_map, None)

        Returns:
            tuple: see compute_loss flag above
        """
        # Step 1: Encode full signal [B, C, T] -> [B, D, T]
        z_encoded = self.encoder(feat)
        B, D, T = z_encoded.shape

        # Step 2: Pool latent at each time step to [B, D] vectors
        # The ODE operates on a single latent vector (the mean across time gives context)
        # For unrolling, we take each time slice z_encoded[:, :, k] as the target
        z_initial = z_encoded[:, :, 0]  # Starting state at t=0: [B, D]

        all_preds   = []  # ODE predictions at each future step
        all_targets = []  # Encoded ground-truth at each future step

        z_current = z_initial
        for k in range(1, min(nsteps + 1, T)):
            # Integrate ODE from t=(k-1) to t=k to predict the next latent state
            z_pred = odeint_rk4(self.dynamics, z_current, t_start=float(k-1),
                                t_end=float(k), n_steps=self.ode_steps)
            z_target = z_encoded[:, :, k]  # Ground-truth at time step k: [B, D]

            all_preds.append(z_pred)
            all_targets.append(z_target)

            # Use the ODE prediction (not the ground truth) for the next step
            # This is the "teacher-forcing-free" continuous unrolling
            z_current = z_pred

        if not compute_loss:
            # Return predicted latent map in [B, D, T] format for compatibility
            T_pred = len(all_preds)
            pred_stack = torch.stack(all_preds, dim=2)  # [B, D, T_pred]
            return pred_stack, None

        # Step 3: Compute JEPA loss
        # Stack all predictions and targets: [B*nsteps, D]
        preds_cat   = torch.cat(all_preds,   dim=0)  # [B * nsteps, D]
        targets_cat = torch.cat(all_targets, dim=0)  # [B * nsteps, D]

        pred_loss   = torch.nn.functional.mse_loss(preds_cat, targets_cat)
        sigreg_loss = self.sigreg(targets_cat)
        total_loss  = pred_loss + self.sigreg_lambda * sigreg_loss

        # Return in the same tuple format as LeWMJEPA.unroll()
        return None, [total_loss, pred_loss, sigreg_loss, {}, None]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Factory & Compatibility Functions (unchanged API for main_jepa_anomaly.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_jepa(in_channels: int = 1, hidden_dim: int = 64, latent_dim: int = 128,
               ode_hidden_dim: int = 256, ode_depth: int = 2, ode_steps: int = 10) -> LatentODE_JEPA:
    """
    Build the Latent ODE JEPA model. Drop-in replacement for the original
    build_jepa() factory in jepa_backbone.py.

    Args:
        in_channels:    number of signal channels (default 2 for acc+severity)
        hidden_dim:     intermediate channels in Conv1D encoder
        latent_dim:     dimension of the latent ODE state space
        ode_hidden_dim: hidden units in the ODEDynamicsNet MLP
        ode_depth:      number of hidden layers in the dynamics MLP
        ode_steps:      RK4 integration steps between each time slice
    """
    encoder  = Encoder1D(in_channels, hidden_dim, latent_dim)
    dynamics = ODEDynamicsNet(latent_dim, ode_hidden_dim, depth=ode_depth)
    model    = LatentODE_JEPA(encoder, dynamics, sigreg_lambda=1.0, ode_steps=ode_steps)
    return model


def build_action_jepa(in_channels: int = 6, action_dim: int = 1,
                      hidden_dim: int = 64, latent_dim: int = 128) -> LatentODE_JEPA:
    """Compatibility alias for action-conditioned JEPA scripts."""
    return build_jepa(in_channels, hidden_dim, latent_dim)


def apply_rankfeat(feat: torch.Tensor) -> torch.Tensor:
    """Remove the rank-1 component from a latent feature map (RankFeat)."""
    B, D, T = feat.size()
    u, s, v = torch.linalg.svd(feat, full_matrices=False)
    rank1_component = s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
    return feat - rank1_component


def apply_rankweight(model: nn.Module) -> None:
    """Strip rank-1 component from all Conv1D/Linear weight matrices (RankWeight)."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            weight = module.weight.data
            original_shape = weight.shape
            if weight.dim() > 2:
                out_channels = original_shape[0]
                weight = weight.view(out_channels, -1)
            u, s, v = torch.linalg.svd(weight, full_matrices=False)
            rank1 = s[0:1].unsqueeze(1) * u[:, 0:1].mm(v[0:1, :])
            weight = weight - rank1
            if len(original_shape) > 2:
                weight = weight.view(*original_shape)
            module.weight.data = weight


def compute_anomaly_score(model: LatentODE_JEPA, data: torch.Tensor,
                           steps: int = 2, use_rankfeat: bool = False) -> float:
    """
    Compute the anomaly score for a single sample as the MSE between the
    ODE-predicted latent trajectory and the encoder's ground-truth latent trajectory.

    A higher score means the signal's dynamics deviate from what the ODE
    learned during self-supervised training — indicative of an anomaly.
    """
    model.eval()
    if data.dim() == 2:
        data = data.unsqueeze(1)

    with torch.no_grad():
        predicted_z_map, _ = model.unroll(data, nsteps=steps, compute_loss=False)
        target_z           = model.encoder(data)        # [B, D, T]

        T_pred = predicted_z_map.shape[2]
        # Align target to the same time range as predictions (t=1 ... t=nsteps)
        target_z = target_z[:, :, 1:T_pred + 1]

        if use_rankfeat:
            target_z      = apply_rankfeat(target_z)
            predicted_z_map = apply_rankfeat(predicted_z_map)

        mse = torch.nn.functional.mse_loss(predicted_z_map, target_z)
    return mse.item()


def jepa_call(model: LatentODE_JEPA, test_data) -> torch.Tensor:
    """Return latent representations for a list of test samples."""
    model.eval()
    representations = []
    with torch.no_grad():
        for data in test_data:
            if isinstance(data, (list, tuple)):
                data = data[0]
            if data.dim() == 2:
                data = data.unsqueeze(1)
            state = model.encoder(data)
            representations.append(state)
    return torch.cat(representations)
