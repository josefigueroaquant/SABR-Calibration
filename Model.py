# ============================================================
# MODEL: Hagan (Black/Normal) + Calibrations in ONE CLASS
# ============================================================
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Dict
from scipy.optimize import least_squares
import Curve
import matplotlib.pyplot as plt



# ========= Hagan formulas =========
def hagan_black_vol(F, K, T, alpha, beta, rho, nu, shift=0.0):
    """
    Compute Black-SABR implied volatility using Hagan's approximation.
    """
    Fp, Kp = F + shift, K + shift
    if Fp <= 0 or Kp <= 0:
        return np.nan
    if np.isclose(Fp, Kp):
        FK = (Fp * Kp) ** ((1.0 - beta) / 2.0)
        term1 = alpha / FK
        term2 = 1.0 + ((1 - beta) ** 2 / 24.0) * (alpha ** 2 / FK ** 2) * T \
                    + (rho * beta * nu * alpha / (4.0 * FK)) * T \
                    + ((2 - 3 * rho ** 2) / 24.0) * (nu ** 2) * T
        return term1 * term2
    logFK = np.log(Fp / Kp)
    z = (nu / alpha) * (Fp * Kp) ** ((1.0 - beta) / 2.0) * logFK
    xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    FK = (Fp * Kp) ** ((1.0 - beta) / 2.0)
    A = alpha / FK
    B = 1.0 + ((1 - beta) ** 2 / 24.0) * (alpha ** 2 / FK ** 2) * T \
            + (rho * beta * nu * alpha / (4.0 * FK)) * T \
            + ((2 - 3 * rho ** 2) / 24.0) * (nu ** 2) * T
    return A * (z / xz) * B


def hagan_normal_vol(F, K, T, alpha, beta, rho, nu, shift=0.0):
    """
    Compute Normal-SABR implied volatility using Hagan's approximation.
    """
    Fp, Kp = F + shift, K + shift
    if np.isclose(Fp, Kp):
        term1 = alpha * (Fp ** (beta - 1.0))
        term2 = 1.0 + ((2 - 3 * rho ** 2) / 24.0) * (nu ** 2) * T \
                    + 0.25 * rho * beta * nu * alpha * (Fp ** (beta - 1.0)) * T \
                    + ((1 - beta) ** 2 / 24.0) * (alpha ** 2) * (Fp ** (2 * beta - 2.0)) * T
        return term1 * term2
    FKb = (Fp ** beta - Kp ** beta) / (beta * (Fp - Kp)) if not np.isclose(Fp, Kp) else Fp ** (beta - 1.0)
    z = (nu / alpha) * (Fp - Kp) * FKb
    xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    A = alpha * (Fp ** (beta - 1.0))
    B = 1.0 + ((2 - 3 * rho ** 2) / 24.0) * (nu ** 2) * T \
            + 0.25 * rho * beta * nu * alpha * (Fp ** (beta - 1.0)) * T \
            + ((1 - beta) ** 2 / 24.0) * (alpha ** 2) * (Fp ** (2 * beta - 2.0)) * T
    return A * (z / xz) * B


def vega_black(F, K, T, sigma, shift=0.0):
    """
    Compute option vega under Black model.
    """
    Fp, Kp = F + shift, K + shift
    if Fp <= 0 or Kp <= 0 or sigma <= 0 or T <= 0:
        return 1.0
    d1 = (np.log(Fp / Kp) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    nprime = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)
    return Fp * nprime * np.sqrt(T)


def vega_normal(F, K, T, sigma):
    """
    Compute option vega under Normal model.
    """
    if sigma <= 0 or T <= 0:
        return 1.0
    d = (F - K) / (sigma * np.sqrt(T))
    nprime = np.exp(-0.5 * d * d) / np.sqrt(2 * np.pi)
    return np.sqrt(T) * nprime


# ========= One-tenor calibrator =========
@dataclass
class SABRParams:
    """
    Container for SABR parameters.
    """
    alpha: float
    beta: float
    rho: float
    nu: float
    shift: float
    model: Literal["black", "normal"]


class SABRCalibratorOneTenor:
    """
    Calibrator for SABR model parameters on a single tenor.
    """
    def __init__(self, beta: float, shift: float, model: Literal["black", "normal"] = "normal"):
        self.beta = beta
        self.shift = shift
        self.model = model

    def _hagan(self, F, K, T, a, r, v):
        return (hagan_black_vol if self.model == "black" else hagan_normal_vol)(F, K, T, a, self.beta, r, v, self.shift)

    def _vega(self, F, K, T, sigma):
        return (vega_black if self.model == "black" else vega_normal)(F, K, T, sigma)

    def _init(self, F, strikes, vols, vol_atm=None):
        """
        Provide initial guesses for SABR parameters.
        """
        if vol_atm is None:
            i_atm = int(np.argmin(np.abs(np.array(strikes) - F)))
            vol_atm = float(vols[i_atm])
        alpha0 = max(1e-8, vol_atm) * (F + self.shift) ** (1.0 - self.beta)
        K = np.array(strikes)
        idx = int(np.argmin(np.abs(K - F)))
        i0 = max(0, idx - 1)
        i1 = min(len(K) - 1, idx + 1)
        try:
            if self.model == "black":
                Fp = F + self.shift
                k0 = np.log((K[i0] + self.shift) / Fp)
                k1 = np.log((K[i1] + self.shift) / Fp)
                skew = (vols[i1] - vols[i0]) / max(1e-10, (k1 - k0))
            else:
                skew = (vols[i1] - vols[i0]) / max(1e-10, (K[i1] - K[i0]))
        except Exception:
            skew = 0.0
        Fp_pow = (F + self.shift) ** (1.0 - self.beta)
        denom = max(1e-8, 0.5 * alpha0 / Fp_pow)
        rho0 = float(np.clip(skew / denom, -0.8, 0.8))
        nu0 = 0.5
        return float(alpha0), float(rho0), float(nu0)

    def calibrate(self, F, T, strikes, vols, *, anchor_atm=True):
        """
        Calibrate SABR parameters for a given tenor.

        Parameters
        ----------
        F : float
            Forward rate.
        T : float
            Time to maturity.
        strikes : array-like
            Strike rates.
        vols : array-like
            Market volatilities.
        anchor_atm : bool
            If True, give higher weight to ATM strike.

        Returns
        -------
        tuple
            (params, fitted_vols, rmse)
        """
        w = np.array([self._vega(F, K, T, max(1e-4, v)) for K, v in zip(strikes, vols)], dtype=float)
        w /= (np.median(w) + 1e-12)
        i_atm = int(np.argmin(np.abs(np.array(strikes) - F)))
        if anchor_atm:
            w[i_atm] *= 10.0

        lb = np.array([1e-8, -0.999, 1e-8])
        ub = np.array([5.0, 0.999, 5.0])
        a0, r0, v0 = self._init(F, strikes, vols)
        x0 = np.clip(np.array([a0, r0, v0], float), lb + 1e-10, ub - 1e-10)

        def resid(theta):
            a, r, v = theta
            m = np.array([self._hagan(F, K, T, a, r, v) for K in strikes])
            return (m - vols) * np.sqrt(w)

        # Single least-squares optimization
        res = least_squares(resid, x0, bounds=(lb, ub),
                            loss="linear",
                            f_scale=0.001,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=8000)

        a, r, v = res.x
        params = SABRParams(a, self.beta, r, v, shift=self.shift, model=self.model)
        fitted = np.array([self._hagan(F, K, T, a, r, v) for K in strikes])
        rmse = float(np.sqrt(np.mean((fitted - vols) ** 2)))
        return params, fitted, rmse

# ========= Beta search =========
def best_beta(F, T, strikes, vols, betas, model="normal", shift=0.0):
    """
    Search for the best beta value by calibrating the SABR model
    across a grid of candidate betas. Selects the beta that gives
    the lowest RMSE error.

    Parameters
    ----------
    F : float
        Forward rate.
    T : float
        Time to maturity.
    strikes : array-like
        Strike prices/rates.
    vols : array-like
        Observed market volatilities.
    betas : array-like
        Candidate beta values to test.
    model : {"normal", "black"}
        Type of SABR model (normal or lognormal).
    shift : float, optional
        Shift applied to rates.

    Returns
    -------
    tuple
        (rmse, params, fitted_vols, beta) for the best beta.
    """
    best = None
    for b in betas:
        cal = SABRCalibratorOneTenor(beta=b, shift=shift, model=model)
        try:
            p, fit, err = cal.calibrate(F, T, strikes, vols, anchor_atm=True)
            if (best is None) or (err < best[0]):
                best = (err, p, fit, b)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Calibration failed for all β values.")
    return best


# ========= Calibrate all tenors =========
def calibrate_all_tenors(df_vols: pd.DataFrame, curve_eur3m: Curve,
                        betas=np.linspace(0.1, 0.9, 9),
                        model="normal", shift=0.0) -> Dict[float, dict]:
    """
    Calibrate SABR parameters for all tenors of a volatility matrix.

    Parameters
    ----------
    df_vols : pandas.DataFrame
        Volatility matrix (rows = tenors, columns = strikes).
    curve_eur3m : Curve
        Discount/fwd curve used to compute forward rates.
    betas : array-like
        Grid of beta values for calibration search.
    model : {"normal", "black"}
        SABR model type.
    shift : float
        Shift parameter for SABR.

    Returns
    -------
    dict
        Dictionary keyed by tenor, each containing calibration results.
    """
    results = {}
    for tenor in df_vols.index:
        T = float(tenor)
        try:
            F = curve_eur3m.fwd_simple(0.0, T)

            # vols in bps -> rate
            vol_row_bps = df_vols.loc[tenor].dropna()
            if vol_row_bps.empty:
                continue
            mkt_vols = vol_row_bps.to_numpy(dtype=float) / 10000.0

            # strikes in % (absolute levels or spreads)
            cols_pct = np.array([float(c) for c in vol_row_bps.index], dtype=float)

            # Heuristic: decide between spreads and levels
            if (np.min(cols_pct) < 0) or (np.ptp(cols_pct) <= 10.0):
                strikes = F + cols_pct / 100.0   # spreads
                mode = "spreads (%)"
            else:
                strikes = cols_pct / 100.0       # absolute levels
                mode = "levels (%)"

            # Cleaning
            mask = np.isfinite(strikes) & np.isfinite(mkt_vols) & (mkt_vols > 0)
            strikes, mkt_vols = strikes[mask], mkt_vols[mask]
            order = np.argsort(strikes)
            strikes, mkt_vols = strikes[order], mkt_vols[order]
            if strikes.size < 3:
                print(f"Tenor {tenor}: insufficient valid strikes")
                continue

            # Best beta calibration
            rmse_b, params_b, fitted_b, beta_b = best_beta(
                F, T, strikes, mkt_vols, betas=betas, model=model, shift=shift
            )

            results[tenor] = {
                "params": params_b, "rmse": rmse_b, "beta": beta_b,
                "F": F, "T": T, "mode": mode,
                "strikes": strikes, "mkt_vols": mkt_vols, "fitted": fitted_b
            }
            print(f"Tenor {tenor}Y -> β={beta_b:.2f}, RMSE={rmse_b:.6e}, mode={mode}")
        except Exception as e:
            print(f"Error calibrating tenor {tenor}: {e}")
            continue
    return results


# ========= Plot smiles =========
def plot_smiles(results: Dict[float, dict], model="normal"):
    """
    Plot SABR implied volatility smiles versus market quotes for each tenor.

    Parameters
    ----------
    results : dict
        Calibration results from calibrate_all_tenors.
    model : {"normal", "black"}
        SABR model type.
    """
    if not results:
        print("No results to plot.")
        return
    items = sorted(results.items(), key=lambda kv: kv[0])
    n = len(items)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for ax, (tenor, res) in zip(axes, items):
        params = res["params"]
        F, T = res["F"], res["T"]
        strikes, mkt_vols = res["strikes"], res["mkt_vols"]
        K_grid = np.linspace(strikes.min(), strikes.max(), 200)
        if model == "black":
            vol_grid = [
                hagan_black_vol(F, K, T, params.alpha, params.beta, params.rho, params.nu, params.shift)
                for K in K_grid
            ]
        else:
            vol_grid = [
                hagan_normal_vol(F, K, T, params.alpha, params.beta, params.rho, params.nu, params.shift)
                for K in K_grid
            ]
        ax.scatter(strikes, mkt_vols, label="Market", zorder=3)
        ax.plot(K_grid, vol_grid, label=f"SABR β={res['beta']:.2f}")
        ax.axvline(F, linestyle="--", color="k", alpha=0.5)
        ax.set_title(f"{tenor}Y  (RMSE={res['rmse']:.2e})")
        ax.set_xlabel("Strike (rate)")
        ax.set_ylabel("Vol (rate)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    for j in range(len(items), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()
