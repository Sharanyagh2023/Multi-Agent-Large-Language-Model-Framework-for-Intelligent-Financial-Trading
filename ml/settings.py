"""Settings of the technical direction model, read from config.yaml -> technical_model.

Every key has a default, so the model also works if the section is missing.
"""
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Large, liquid Nifty 50 stocks with long price histories. Used only to TRAIN the
# model; any NSE/BSE stock can be analysed. (Survivorship bias: today's large caps.)
DEFAULT_UNIVERSE: Tuple[str, ...] = (
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "SBIN.NS",
    "ITC.NS", "HINDUNILVR.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "MARUTI.NS", "M&M.NS", "SUNPHARMA.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "ASIANPAINT.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "GRASIM.NS", "ADANIPORTS.NS", "ADANIENT.NS",
    "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "BRITANNIA.NS", "TATACONSUM.NS", "SBILIFE.NS", "HDFCLIFE.NS", "TRENT.NS", "BEL.NS", "SHRIRAMFIN.NS",
)


@dataclass(frozen=True)
class TechnicalModelSettings:
    horizon_days: int = 5                     # predict the move over the next N trading days
    flat_threshold_vol_multiple: float = 0.5  # |move| below this many "normal moves" counts as flat
    training_years: int = 8                   # history used for training
    max_model_age_days: int = 90              # retrain when the newest saved model is older than this
    min_edge: float = 0.03                    # balanced accuracy above chance for "weak"; 2x for "useful"
    validation_folds: int = 4                 # walk-forward test blocks
    validation_share: float = 0.4             # most recent share of dates used for walk-forward testing
    calibration_share: float = 0.15           # most recent share of training dates used to calibrate probabilities
    top_drivers: int = 5                      # SHAP reasons reported to the LLM
    min_training_rows: int = 5000
    artifacts_dir: Path = PROJECT_ROOT / "ml" / "artifacts"
    universe: Tuple[str, ...] = field(default=DEFAULT_UNIVERSE)

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def price_cache_dir(self) -> Path:
        return self.artifacts_dir / "price_cache"

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]]) -> "TechnicalModelSettings":
        values = dict(values or {})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise KeyError(f"Unknown technical_model settings in config.yaml: {sorted(unknown)}")
        if "artifacts_dir" in values:
            path = Path(values["artifacts_dir"])
            values["artifacts_dir"] = path if path.is_absolute() else PROJECT_ROOT / path
        if "universe" in values:
            values["universe"] = tuple(values["universe"])
        return cls(**values)

    @classmethod
    def from_config_file(cls, path: Path = CONFIG_PATH) -> "TechnicalModelSettings":
        with open(path, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        return cls.from_dict(raw.get("technical_model"))
