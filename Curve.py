import pandas as pd
import numpy as np
import math
from datetime import datetime
from dataclasses import dataclass
from typing import Iterable, Tuple


def year_fraction(d0, d1, basis: str = "ACT/360") -> float:
    """
    Compute the year fraction between two dates using a given day-count basis.

    Parameters
    ----------
    d0 : datetime-like
        Start date.
    d1 : datetime-like
        End date.
    basis : {"ACT/360", "ACT/365"}
        Day-count convention.

    Returns
    -------
    float
        Year fraction between d0 and d1 under the selected basis.
    """
    d0 = pd.to_datetime(d0)
    d1 = pd.to_datetime(d1)
    days = (d1 - d0).days
    if basis.upper() == "ACT/365":
        return days / 365.0
    if basis.upper() == "ACT/360":
        return days / 360.0
    raise ValueError("basis not found")


def to_years(d, val_date) -> float:
    """
    Convert a date to a year fraction from a valuation date using ACT/360.

    Parameters
    ----------
    d : datetime-like
        Target date.
    val_date : datetime-like
        Valuation/origin date.

    Returns
    -------
    float
        Time in years from val_date to d under ACT/360.
    """
    return year_fraction(val_date, pd.to_datetime(d).normalize(), "ACT/360")


@dataclass
class Curve:
    """
    Log-linear discount factor curve with simple forward extraction.
    """
    times: np.ndarray  # times in years
    dfs:   np.ndarray  # discount factors

    @staticmethod
    def from_pillars(pillars: Iterable[Tuple[float, float]]) -> "Curve":
        """
        Build a Curve from (time, df) pillars.

        Parameters
        ----------
        pillars : Iterable[Tuple[float, float]]
            Sequence of (time_in_years, discount_factor) points.

        Returns
        -------
        Curve
            Interpolable discount factor curve.
        """
        pts = sorted(pillars)
        t = np.array([p[0] for p in pts], dtype=float)
        d = np.array([p[1] for p in pts], dtype=float)
        return Curve(t, d)

    @staticmethod
    def read_volatility_matrix(file_path: str, sheet_name: str = "VCUB") -> pd.DataFrame:
        """
        Read a volatility matrix from an Excel sheet where the first column
        contains tenors (as numeric years) and the header row contains strikes.

        The function attempts to detect the header row by looking for keywords
        like 'Plazo/Strike', 'Plazo', or 'Strike' in the first row that contains them.

        Parameters
        ----------
        file_path : str
            Path to the Excel file.
        sheet_name : str, optional
            Sheet name to read from, by default "VCUB".

        Returns
        -------
        pandas.DataFrame
            DataFrame with index=tenors (float years) and columns=strikes (strings as found).
        """
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        header_row = None
        for i, row in df_raw.iterrows():
            if any(str(c).strip() in ['Plazo/Strike', 'Plazo', 'Strike'] for c in row if pd.notna(c)):
                header_row = i
                break
        if header_row is None:
            header_row = 0
        headers = df_raw.iloc[header_row, 1:].astype(str).tolist()
        data_start = header_row + 1
        tenors_col = df_raw.iloc[data_start:, 0]
        tenors = [float(x) for x in tenors_col if pd.notna(x)]   # tenors in years (float)
        mat = df_raw.iloc[data_start:data_start + len(tenors), 1:1 + len(headers)]
        mat = mat.apply(pd.to_numeric, errors='coerce')
        df = pd.DataFrame(mat.to_numpy(), index=tenors, columns=headers)
        df.index.name = "Tenor"
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        return df

    def _interp_log_df(self, t: float) -> float:
        """
        Log-linear interpolation of discount factors.

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Interpolated discount factor at time t.
        """
        if t <= self.times[0]:
            return float(self.dfs[0])
        if t >= self.times[-1]:
            return float(self.dfs[-1])
        i = np.searchsorted(self.times, t)
        t0, t1 = self.times[i - 1], self.times[i]
        y0, y1 = math.log(self.dfs[i - 1]), math.log(self.dfs[i])
        w = (t - t0) / (t1 - t0)
        return math.exp(y0 + w * (y1 - y0))

    def df(self, t: float) -> float:
        """
        Get the discount factor at time t (years) using log-linear interpolation.

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Discount factor at t.
        """
        return self._interp_log_df(t)

    def fwd_simple(self, t0: float, t1: float) -> float:
        """
        Compute the simple forward rate between times t0 and t1 (in years).

        Parameters
        ----------
        t0 : float
            Start time in years.
        t1 : float
            End time in years.

        Returns
        -------
        float
            Simple forward rate over [t0, t1].
        """
        p0, p1 = self.df(t0), self.df(t1)
        alpha = max(1e-12, (t1 - t0))
        return (p0 / p1 - 1.0) / alpha


if __name__ == "__main__":
    # === INPUT FILE PATH ===
    xls_path = r"YourPATH\Datos.xlsx"

    # === READ DISCOUNT FACTORS SHEET ===
    df = pd.read_excel(xls_path, sheet_name="FD", header=1)

    # Rename columns to consistent names (as found in the file)
    df = df.rename(columns={
        "Date":   "Date_ESTR",
        "FD":     "DF_ESTR",
        "Date.1": "Date_EUR3M",
        "FD.1":   "DF_EUR3M",
    })

    # Parse dates
    for c in ["Date_ESTR", "Date_EUR3M"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")

    # Parse numeric discount factors
    for c in ["DF_ESTR", "DF_EUR3M"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Extract valid rows for each curve
    estr = df.loc[~df["Date_ESTR"].isna() & ~df["DF_ESTR"].isna(), ["Date_ESTR", "DF_ESTR"]].copy()
    eur3m = df.loc[~df["Date_EUR3M"].isna() & ~df["DF_EUR3M"].isna(), ["Date_EUR3M", "DF_EUR3M"]].copy()

    # Sort by date
    estr = estr.sort_values("Date_ESTR").reset_index(drop=True)
    eur3m = eur3m.sort_values("Date_EUR3M").reset_index(drop=True)

    # Valuation date = earliest available date across both curves
    val_date = min(estr.loc[0, "Date_ESTR"], eur3m.loc[0, "Date_EUR3M"]).normalize()

    # Times in years from valuation date
    t_estr  = np.array([to_years(d, val_date) for d in estr["Date_ESTR"]], dtype=float)
    t_eur3m = np.array([to_years(d, val_date) for d in eur3m["Date_EUR3M"]], dtype=float)

    # Remove duplicate times, keep first occurrence indices
    t_estr_u,  idx_e = np.unique(t_estr,  return_index=True)
    t_eur3m_u, idx_b = np.unique(t_eur3m, return_index=True)

    # Build discount curves from (time, df) pillars
    curve_estr  = Curve.from_pillars(list(zip(t_estr_u,  estr["DF_ESTR"].to_numpy()[idx_e])))
    curve_eur3m = Curve.from_pillars(list(zip(t_eur3m_u, eur3m["DF_EUR3M"].to_numpy()[idx_b])))

    # === VOLATILITIES ===
    df_volatilities = Curve.read_volatility_matrix(xls_path, "VCUB")
    print("=== VOLATILITY MATRIX ===")
    print(f"Shape: {df_volatilities.shape}")
    print(f"Index (Tenors): {list(df_volatilities.index)}")
    print(f"Columns (Strikes): {list(df_volatilities.columns)}")
    print("\nFirst rows:")
    print(df_volatilities)
