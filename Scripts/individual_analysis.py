#!/usr/bin/env python3
"""
individual_analysis.py
================================
Individual Report — RQ1 & RQ2  [Evan Morel]

RQ1: How does the plywood price-S&P 500 relationship reflect broader
     housing market health vs manufacturing & supply chain conditions?

RQ2: Does combining plywood + steel PPI into a Construction Cost Index (CCI)
     improve predictions over plywood alone?

New datasets (beyond group project):
  WPU101.csv    — Iron & Steel PPI, FRED (1926-2026)
  HOUST.csv     — Housing Starts, FRED (1959-2026)
  IPMANSICS.csv — Manufacturing Output Index SIC SA, FRED (1919-2026)

Technique: OLS regression stratified by economic regime +
           36-month rolling correlation comparison (CCI vs plywood alone)


"""

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

# colour palette
C_PLY   = "#C0392B"
C_HOUST = "#2980B9"
C_IPMAN = "#27AE60"
C_STEEL = "#8E44AD"
C_SP5   = "#2C3E6B"
C_DMND  = "#2980B9"
C_SUPP  = "#E74C3C"
C_NEUT  = "#95A5A6"


# ===========================================================================
# DATA LOADER
# ===========================================================================

class DataLoader:
    """
    Loads and merges all five datasets.

    Regime classification (RQ1)
    ---------------------------
    Demand-Pull : housing starts YoY > +5%  AND  mfg output YoY > +1%
                  plywood price rises driven by construction demand
    Supply-Push : housing starts YoY < 0    AND  mfg output YoY < 0
                  plywood price rises despite weak demand (cost-push)
    Neutral     : everything else
    """

    def __init__(self, sp500_path, plywood_path, steel_path, houst_path, ipman_path):
        self.paths = dict(
            sp500=Path(sp500_path), plywood=Path(plywood_path),
            steel=Path(steel_path), houst=Path(houst_path), ipman=Path(ipman_path),
        )

    def _load_fred(self, path, col):
        df = pd.read_csv(path, parse_dates=["observation_date"])
        df = df.rename(columns={"observation_date": "date", col: col})
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        return df.sort_values("date").reset_index(drop=True)

    def build(self) -> pd.DataFrame:
        plywood = self._load_fred(self.paths["plywood"], "WPU083")
        steel = self._load_fred(self.paths["steel"], "WPU101")
        houst = self._load_fred(self.paths["houst"], "HOUST")
        ipman = self._load_fred(self.paths["ipman"], "IPMANSICS")

        sp500 = pd.read_csv(self.paths["sp500"], parse_dates=["Date"])
        sp500 = sp500.rename(columns={"Date": "date", "Close": "sp500"})
        sp500_m = (sp500.set_index("date").resample("MS")
                   .last()[["sp500"]].reset_index())

        merged = (plywood
                  .merge(steel, on="date", how="inner")
                  .merge(houst, on="date", how="left")
                  .merge(ipman, on="date", how="left")
                  .merge(sp500_m, on="date", how="inner")
                  )

        # year-over-year returns
        merged["ply_yoy"] = merged["WPU083"].pct_change(12)
        merged["steel_yoy"] = merged["WPU101"].pct_change(12)
        merged["houst_yoy"] = merged["HOUST"].pct_change(12)
        merged["ipman_yoy"] = merged["IPMANSICS"].pct_change(12)
        merged["sp500_yoy"] = merged["sp500"].pct_change(12)
        merged["sp500_fwd3"] = merged["sp500"].pct_change(3).shift(-3)

        # Composite Construction Cost Index — equal-weight z-score (RQ2)
        for col in ["WPU083", "WPU101"]:
            mu = merged[col].mean();
            sd = merged[col].std()
            merged[f"{col}_z"] = (merged[col] - mu) / sd
        merged["CCI"] = (merged["WPU083_z"] + merged["WPU101_z"]) / 2
        merged["cci_yoy"] = merged["CCI"].pct_change(12)

        # Regime classification (RQ1)
        dp = ((merged["houst_yoy"] > 0.05) & (merged["ipman_yoy"] > 0.01))
        sp = ((merged["houst_yoy"] < 0) & (merged["ipman_yoy"] < 0))
        merged["demand_pull"] = dp.astype(int)
        merged["supply_push"] = sp.astype(int)
        merged["regime"] = "Neutral"
        merged.loc[dp, "regime"] = "Demand-Pull"
        merged.loc[sp, "regime"] = "Supply-Push"

        self.merged = merged
        clean = merged.dropna(subset=["ply_yoy", "houst_yoy", "ipman_yoy", "sp500_fwd3"])
        print(f"[DataLoader] {len(clean)} months  "
              f"{clean['date'].min().date()} to {clean['date'].max().date()}")
        print("[DataLoader] Regime counts:")
        print(clean["regime"].value_counts().to_string())
        return merged


# ===========================================================================
# STATISTICAL ANALYSIS
# ===========================================================================

class IndividualAnalysis:
    """
    RQ1: OLS of plywood YoY -> S&P 500 fwd 3M return, per economic regime.
         Compares slope and r across Demand-Pull, Neutral, Supply-Push.

    RQ2: 36-month rolling Pearson |r| of plywood alone vs CCI with
         S&P 500 YoY return. Reports mean |r| over full history.
    """

    def __init__(self, merged: pd.DataFrame):
        self.merged = merged
        self.clean = merged.dropna(
            subset=["ply_yoy", "houst_yoy", "ipman_yoy", "sp500_fwd3"]
        )

    def rq1_regime_ols(self) -> pd.DataFrame:
        print("\n" + "=" * 55)
        print("RQ1 - OLS BY ECONOMIC REGIME")
        print("=" * 55)
        print("Dependent  : S&P 500 3-month forward return")
        print("Predictor  : Plywood PPI year-over-year % change")
        print("Regimes    : Demand-Pull / Neutral / Supply-Push\n")
        rows = []
        for regime in ["Demand-Pull", "Neutral", "Supply-Push"]:
            sub = self.clean[self.clean["regime"] == regime]
            x, y = sub["ply_yoy"].values, sub["sp500_fwd3"].values
            sl, ic, r, p, _ = stats.linregress(x, y)
            rows.append(dict(Regime=regime, n=len(sub),
                             r=round(r, 4), R2=round(r ** 2, 5),
                             slope=round(sl, 5), p_value=round(p, 4)))
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print("\nKey: slope sign flips between Demand-Pull and Supply-Push")
        return df

    def rq2_cci_vs_plywood(self) -> None:
        print("\n" + "=" * 55)
        print("RQ2 - CCI COMPOSITE vs PLYWOOD ALONE")
        print("=" * 55)
        data = self.merged.dropna(subset=["ply_yoy", "cci_yoy", "sp500_yoy"])
        roll_ply = data["ply_yoy"].rolling(36).corr(data["sp500_yoy"])
        roll_cci = data["cci_yoy"].rolling(36).corr(data["sp500_yoy"])
        mean_ply = roll_ply.abs().mean()
        mean_cci = roll_cci.abs().mean()
        winner = "CCI" if mean_cci > mean_ply else "Plywood alone"
        print(f"Mean |rolling r|  Plywood alone : {mean_ply:.4f}")
        print(f"Mean |rolling r|  CCI composite : {mean_cci:.4f}")
        print(f"Stronger predictor overall      : {winner}")
        cci_better = (roll_cci.abs() > roll_ply.abs()).sum()
        total = roll_cci.abs().notna().sum()
        print(f"Months where CCI |r| > Plywood  : {cci_better}/{total} ({100 * cci_better / total:.1f}%)")

    def run_all(self) -> None:
        self.rq1_regime_ols()
        self.rq2_cci_vs_plywood()


# ===========================================================================
# VISUALISATIONS
# ===========================================================================

class IndividualVisualizations:
    """
    fig1_overview       : 4-panel time series, Supply-Push shading
    fig2_decomposition  : Scatter triptych coloured by regime
    fig3_regime_ols     : OLS per regime - does context change the signal?
    fig4_cci_vs_plywood : Rolling correlation: CCI vs plywood alone (RQ2)
    """

    def __init__(self, merged: pd.DataFrame, out_dir=".", show=False):
        self.merged = merged
        self.clean = merged.dropna(
            subset=["ply_yoy", "houst_yoy", "ipman_yoy", "sp500_fwd3"]
        )
        self.out_dir = Path(out_dir)
        self.show = show
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, fname):
        path = self.out_dir / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
        if self.show:
            plt.show()
        plt.close(fig)

    def _shade_supply_push(self, ax, data):
        """Shade Supply-Push periods on any axis."""
        sp_mask = data["regime"] == "Supply-Push"
        in_sp = False;
        start = None
        prev_date = None
        for _, row in data.iterrows():
            if sp_mask.loc[row.name] and not in_sp:
                start = row["date"];
                in_sp = True
            elif not sp_mask.loc[row.name] and in_sp:
                ax.axvspan(start, row["date"], alpha=0.10, color=C_SUPP)
                in_sp = False

    # -----------------------------------------------------------------------
    def fig1_overview(self) -> None:
        """
        Four-panel time series of all datasets. Supply-Push regimes shaded red.
        Shows the viewer the full data landscape before the analysis begins.
        """
        data = self.merged.dropna(subset=["WPU083", "HOUST", "IPMANSICS", "sp500"])
        panels = [
            ("WPU083", "Plywood PPI (WPU083)", C_PLY),
            ("HOUST", "Housing Starts (thousands, SAAR)", C_HOUST),
            ("IPMANSICS", "Manufacturing Output (IPMANSICS)", C_IPMAN),
            ("sp500", "S&P 500 Close (log scale)", C_SP5),
        ]
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        for ax, (col, label, color) in zip(axes, panels):
            ax.plot(data["date"], data[col], color=color, lw=1.4, alpha=0.9)
            ax.fill_between(data["date"], data[col], alpha=0.07, color=color)
            ax.set_ylabel(label, fontsize=9, color=color)
            ax.tick_params(axis="y", labelcolor=color, labelsize=8)
            ax.grid(axis="y", alpha=0.2)
            self._shade_supply_push(ax, data)
        axes[3].set_yscale("log")
        axes[3].xaxis.set_major_locator(mdates.YearLocator(10))
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(axes[3].get_xticklabels(), rotation=30, fontsize=9)
        patch = mpatches.Patch(facecolor=C_SUPP, alpha=0.3,
                               label="Supply-Push Regime (housing down + mfg down)")
        axes[0].legend(handles=[patch], loc="upper left", fontsize=9, framealpha=0.9)
        fig.suptitle(
            "All Datasets: Plywood PPI, Housing Starts, Manufacturing Output, S&P 500\n"
            "(1960-2025, monthly  |  red shading = Supply-Push regimes)",
            fontsize=13, fontweight="bold")
        fig.text(0.5, -0.01,
                 "Figure 1 | Overview of all four datasets on a common time axis. "
                 "Red-shaded bands = Supply-Push regimes: months where both "
                 "housing starts and manufacturing output contracted simultaneously.",
                 ha="center", fontsize=9, style="italic")
        plt.tight_layout()
        self._save(fig, "rq1_fig1_overview.png")

    # -----------------------------------------------------------------------
    def fig2_decomposition(self) -> None:
        """
        Scatter triptych: plywood vs housing starts, mfg output, S&P 500 fwd.
        Points coloured by regime to show demand-pull vs supply-push separation.
        """
        palette = {"Demand-Pull": C_DMND, "Supply-Push": C_SUPP, "Neutral": C_NEUT}
        markers = {"Demand-Pull": "o", "Supply-Push": "s", "Neutral": "^"}
        comparisons = [
            ("houst_yoy", "Housing Starts YoY %",
             "Plywood vs Housing Starts\n(Demand-Pull signal)"),
            ("ipman_yoy", "Manufacturing Output YoY %",
             "Plywood vs Manufacturing Output\n(Supply-Push signal)"),
            ("sp500_fwd3", "S&P 500 Forward 3-Month Return",
             "Plywood vs S&P 500 Fwd Return\n(coloured by regime)"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        for ax, (ycol, ylabel, title) in zip(axes, comparisons):
            sub = self.clean.dropna(subset=["ply_yoy", ycol])
            for regime, grp in sub.groupby("regime"):
                ax.scatter(grp["ply_yoy"], grp[ycol],
                           color=palette[regime], marker=markers[regime],
                           alpha=0.45, s=18, label=regime, edgecolors="none")
            x = sub["ply_yoy"].values;
            y = sub[ycol].values
            sl, ic, r, p, _ = stats.linregress(x, y)
            xfit = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 200)
            ax.plot(xfit, sl * xfit + ic, color="black", lw=2,
                    label=f"OLS  r={r:.3f}  p={p:.3f}")
            ax.axhline(0, color="grey", lw=0.7, ls="--")
            ax.axvline(0, color="grey", lw=0.7, ls="--")
            ax.set_xlabel("Plywood PPI YoY % Change", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8.5, markerscale=1.4, framealpha=0.9)
            ax.grid(alpha=0.15)
        fig.suptitle(
            "RQ1 - Plywood Price Decomposed:\n"
            "Housing Market Health vs Manufacturing / Supply Chain Conditions",
            fontsize=13, fontweight="bold", y=1.01)
        fig.text(0.5, -0.04,
                 "Figure 2 | Each point = one month (1960-2025). "
                 "Blue circles = Demand-Pull (housing up + mfg up), "
                 "Red squares = Supply-Push (housing down + mfg down), Grey = Neutral.\n"
                 "Left: plywood tracks housing demand. Centre: plywood vs industrial output. "
                 "Right: does regime change plywood's S&P 500 predictive power?",
                 ha="center", fontsize=9, style="italic")
        plt.tight_layout()
        self._save(fig, "rq1_fig2_decomposition.png")

    # -----------------------------------------------------------------------
    def fig3_regime_ols(self) -> None:
        """
        OLS of plywood YoY -> S&P 500 fwd 3M return, one panel per regime.
        Central test of RQ1: does economic context moderate the signal?
        """
        regimes = [("Demand-Pull", C_DMND), ("Neutral", C_NEUT), ("Supply-Push", C_SUPP)]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
        for ax, (regime, color) in zip(axes, regimes):
            sub = self.clean[self.clean["regime"] == regime].dropna(
                subset=["ply_yoy", "sp500_fwd3"])
            x, y = sub["ply_yoy"].values, sub["sp500_fwd3"].values
            ax.scatter(x, y, alpha=0.35, s=20, color=color, edgecolors="none")
            sl, ic, r, p, _ = stats.linregress(x, y)
            xfit = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 200)
            ax.plot(xfit, sl * xfit + ic, color="black", lw=2.2,
                    label=f"r = {r:.3f}\np = {p:.3f}\nn = {len(sub)}")
            ax.axhline(0, color="grey", lw=0.7, ls="--")
            ax.axvline(0, color="grey", lw=0.7, ls="--")
            ax.set_title(f"{regime} Regime", fontsize=12, fontweight="bold", color=color)
            ax.set_xlabel("Plywood PPI YoY % Change", fontsize=10)
            ax.legend(fontsize=10, framealpha=0.9)
            ax.grid(alpha=0.15)
        axes[0].set_ylabel("S&P 500 Forward 3-Month Return", fontsize=10)
        fig.suptitle(
            "RQ1 - Does Housing vs Supply Chain Context Moderate\n"
            "Plywood's Relationship with S&P 500 Forward Returns?",
            fontsize=12, fontweight="bold", y=1.01)
        fig.text(0.5, -0.04,
                 "Figure 3 | OLS of plywood YoY on S&P 500 3-month forward return per regime.\n"
                 "Demand-Pull = housing starts up + mfg output up. "
                 "Supply-Push = housing starts down + mfg output down. "
                 "Slope sign, r, and p-value differences show whether regime moderates the signal.",
                 ha="center", fontsize=9, style="italic")
        plt.tight_layout()
        self._save(fig, "rq1_fig3_regime_ols.png")

    # -----------------------------------------------------------------------
    def fig4_cci_vs_plywood(self) -> None:
        """
        RQ2: 36-month rolling |r| of plywood alone vs CCI (plywood+steel)
        with S&P 500 YoY return. Shows whether combining materials improves signal.
        """
        data = self.merged.dropna(subset=["ply_yoy", "cci_yoy", "sp500_yoy"])
        roll_ply = data["ply_yoy"].rolling(36).corr(data["sp500_yoy"])
        roll_cci = data["cci_yoy"].rolling(36).corr(data["sp500_yoy"])
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        axes[0].plot(data["date"], roll_ply, color=C_PLY, lw=1.8,
                     label="Plywood alone (WPU083)")
        axes[0].plot(data["date"], roll_cci, color=C_STEEL, lw=1.8, ls="--",
                     label="CCI composite (Plywood + Steel, equal weight)")
        axes[0].axhline(0, color="grey", lw=0.7, ls=":")
        axes[0].fill_between(data["date"], roll_cci, roll_ply,
                             where=(roll_cci.abs() > roll_ply.abs()),
                             alpha=0.15, color=C_STEEL)
        axes[0].fill_between(data["date"], roll_cci, roll_ply,
                             where=(roll_cci.abs() <= roll_ply.abs()),
                             alpha=0.15, color=C_PLY)
        axes[0].set_ylabel("36-Month Rolling Pearson r\nvs S&P 500 YoY", fontsize=10)
        axes[0].set_title(
            "RQ2 - CCI Composite vs Plywood Alone: 36-Month Rolling Correlation with S&P 500",
            fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=10, framealpha=0.9)
        axes[0].grid(alpha=0.2)

        ply_n = ((self.merged["WPU083"] - self.merged["WPU083"].mean())
                 / self.merged["WPU083"].std())
        axes[1].plot(self.merged["date"], ply_n, color=C_PLY, lw=1.4,
                     alpha=0.85, label="Plywood PPI (z-score)")
        axes[1].plot(self.merged["date"], self.merged["CCI"], color=C_STEEL,
                     lw=1.4, alpha=0.85, ls="--", label="CCI (z-score)")
        axes[1].set_ylabel("Standardised Level (z-score)", fontsize=10)
        axes[1].set_title("Plywood PPI vs CCI - Standardised Price Levels", fontsize=11)
        axes[1].legend(fontsize=10, framealpha=0.9)
        axes[1].grid(alpha=0.2)
        axes[1].xaxis.set_major_locator(mdates.YearLocator(10))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(axes[1].get_xticklabels(), rotation=30)
        fig.text(0.5, -0.02,
                 "Figure 4 | TOP: 36-month rolling Pearson r of plywood alone (red) and "
                 "CCI composite (purple dashed) with S&P 500 YoY return.\n"
                 "Purple shading = CCI stronger; red shading = plywood alone stronger. "
                 "BOTTOM: both series standardised for visual comparison.",
                 ha="center", fontsize=9, style="italic")
        plt.tight_layout()
        self._save(fig, "rq2_fig4_cci_vs_plywood.png")

    def run_all(self) -> None:
        print("\n[IndividualVisualizations] Generating figures...")
        self.fig1_overview()
        self.fig2_decomposition()
        self.fig3_regime_ols()
        self.fig4_cci_vs_plywood()
        print("[IndividualVisualizations] Done.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    print("S&P 500 vs Plywood — Housing & Manufacturing Analysis")
    print("=" * 55)

    # CSVs live in the same folder as this script
    script_dir = Path(__file__).parent

    loader = DataLoader(
        sp500_path=script_dir / "sp500_daily.csv",
        plywood_path=script_dir / "WPU083.csv",
        steel_path=script_dir / "WPU101.csv",
        houst_path=script_dir / "HOUST.csv",
        ipman_path=script_dir / "IPMANSICS.csv",
    )
    merged = loader.build()

    # Run statistical analysis
    analysis = IndividualAnalysis(merged)
    analysis.run_all()

    # Generate and save all figures
    viz = IndividualVisualizations(merged, out_dir=script_dir, show=False)
    viz.run_all()

    print("\n" + "=" * 55)
    print("Analysis complete!")
    print(f"Figures saved to: {script_dir.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()
