"""Prepare deterministic local datasets for quick-reference notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
import pandas as pd

SEED = 42
DEFAULT_OUT_DIR = Path("data")

AMES_PATH = DEFAULT_OUT_DIR / "ames_housing.csv"
TITANIC_PATH = DEFAULT_OUT_DIR / "titanic.csv"
PENGUINS_PATH = DEFAULT_OUT_DIR / "penguins.csv"
BIKE_PATH = DEFAULT_OUT_DIR / "bike_sharing.csv"
LALONDE_PATH = DEFAULT_OUT_DIR / "lalonde.csv"
CPS_PANEL_PATH = DEFAULT_OUT_DIR / "cps_panel.csv"
SOURCES_MANIFEST_CANONICAL = DEFAULT_OUT_DIR / "quick_reference_sources.json"
SOURCES_MANIFEST_COMPAT = DEFAULT_OUT_DIR / "phase35_sources.json"

LALONDE_URL = (
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MatchIt/lalonde.csv"
)
CPS_PANEL_URL = (
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/DRDID/nsw_long.csv"
)
SNAPSHOT_TRANSFORM_VERSION = "quickref-v1"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_ames_housing(seed: int) -> pd.DataFrame:
    src = Path("examples/data/california_housing.csv")
    if not src.exists():
        raise FileNotFoundError(f"Missing source dataset: {src}")

    rng = np.random.default_rng(seed)
    base = pd.read_csv(src)
    take_n = min(len(base), 5000)
    frame = base.sample(n=take_n, random_state=seed).reset_index(drop=True)

    gr_liv_area = np.clip(
        frame["AveRooms"] * 420.0 + rng.normal(0.0, 90.0, size=take_n), 400.0, None
    )
    lot_area = np.clip(
        frame["Population"] * 18.0 + rng.normal(0.0, 250.0, size=take_n), 2000.0, None
    )
    overall_qual = np.clip(np.round(frame["MedInc"] * 1.8 + 2.0), 1, 10).astype(int)
    overall_cond = np.clip(np.round(10.0 - frame["AveBedrms"] * 4.0), 1, 10).astype(int)
    year_built = np.clip(1950 + np.round(frame["HouseAge"]), 1950, 2020).astype(int)
    garage_cars = np.clip(np.round(frame["AveOccup"]), 1, 4).astype(int)

    neighborhood = pd.cut(
        frame["Latitude"],
        bins=[-np.inf, 34.5, 36.0, 37.5, np.inf],
        labels=["South", "Central", "North", "Hill"],
    ).astype(str)
    house_style = pd.cut(
        frame["AveRooms"],
        bins=[-np.inf, 4.5, 6.0, 7.5, np.inf],
        labels=["1Story", "2Story", "Split", "Loft"],
    ).astype(str)
    bldg_type = pd.cut(
        frame["AveOccup"],
        bins=[-np.inf, 2.0, 3.0, 4.0, np.inf],
        labels=["Fam", "Town", "Duplex", "Multi"],
    ).astype(str)

    sale_price = (
        40_000.0
        + 65_000.0 * frame["MedInc"].to_numpy(dtype=float)
        + 22.0 * gr_liv_area
        + 1.8 * lot_area
        + 8_000.0 * overall_qual
        + 3_000.0 * garage_cars
        + rng.normal(0.0, 25_000.0, size=take_n)
    )
    sale_price = np.clip(sale_price, 50_000.0, None)

    out = pd.DataFrame(
        {
            "LotArea": lot_area.round(2),
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "YearBuilt": year_built,
            "GrLivArea": gr_liv_area.round(2),
            "GarageCars": garage_cars,
            "Neighborhood": neighborhood,
            "HouseStyle": house_style,
            "BldgType": bldg_type,
            "SalePrice": sale_price.round(2),
        }
    )
    return out


def generate_titanic(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_rows = 891

    pclass = rng.choice([1, 2, 3], size=n_rows, p=[0.25, 0.22, 0.53])
    sex = rng.choice(["female", "male"], size=n_rows, p=[0.35, 0.65])
    age = np.clip(rng.normal(31.0, 14.0, size=n_rows), 0.42, 80.0)
    sib_sp = np.clip(rng.poisson(0.7, size=n_rows), 0, 6)
    parch = np.clip(rng.poisson(0.5, size=n_rows), 0, 5)
    embarked = rng.choice(["S", "C", "Q"], size=n_rows, p=[0.72, 0.2, 0.08])
    cabin_deck = rng.choice(["A", "B", "C", "D", "E", "F", "G", "U"], size=n_rows)

    fare_base = np.select(
        [pclass == 1, pclass == 2, pclass == 3],
        [85.0, 30.0, 13.0],
        default=20.0,
    )
    fare = np.clip(fare_base + rng.normal(0.0, 12.0, size=n_rows) + sib_sp * 2.0, 4.0, 300.0)

    logits = (
        1.4 * (sex == "female").astype(float)
        - 0.75 * (pclass == 3).astype(float)
        - 0.35 * (pclass == 2).astype(float)
        - 0.03 * age
        - 0.12 * sib_sp
        - 0.08 * parch
        + 0.004 * fare
        + 0.2 * (embarked == "C").astype(float)
        + rng.normal(0.0, 0.35, size=n_rows)
    )
    survived = rng.binomial(1, _sigmoid(logits), size=n_rows)

    out = pd.DataFrame(
        {
            "PassengerId": np.arange(1, n_rows + 1),
            "Pclass": pclass,
            "Sex": sex,
            "Age": age.round(2),
            "SibSp": sib_sp,
            "Parch": parch,
            "Fare": fare.round(2),
            "Embarked": embarked,
            "CabinDeck": cabin_deck,
            "Survived": survived,
        }
    )
    return out


def generate_penguins(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    species_levels = ["Adelie", "Chinstrap", "Gentoo"]
    sizes = [152, 68, 124]

    species = np.concatenate(
        [np.repeat(species_levels[idx], size) for idx, size in enumerate(sizes)]
    )
    n_rows = int(species.shape[0])
    rng.shuffle(species)

    island_map = {
        "Adelie": ["Torgersen", "Biscoe", "Dream"],
        "Chinstrap": ["Dream"],
        "Gentoo": ["Biscoe"],
    }
    sex = rng.choice(["male", "female"], size=n_rows, p=[0.5, 0.5])

    means = {
        "Adelie": (38.8, 18.3, 190.0, 3700.0),
        "Chinstrap": (48.8, 18.4, 196.0, 3730.0),
        "Gentoo": (47.5, 14.9, 217.0, 5070.0),
    }

    rows: list[dict[str, float | str]] = []
    for idx, sp in enumerate(species):
        mu_bl, mu_bd, mu_fl, mu_bm = means[str(sp)]
        island = rng.choice(island_map[str(sp)])
        sex_shift = 1.4 if sex[idx] == "male" else -1.1
        rows.append(
            {
                "bill_length_mm": float(np.clip(rng.normal(mu_bl + sex_shift, 2.8), 30.0, 60.0)),
                "bill_depth_mm": float(
                    np.clip(rng.normal(mu_bd + 0.25 * sex_shift, 1.2), 12.0, 22.0)
                ),
                "flipper_length_mm": float(
                    np.clip(rng.normal(mu_fl + 2.0 * sex_shift, 7.0), 170.0, 235.0)
                ),
                "body_mass_g": float(
                    np.clip(rng.normal(mu_bm + 60.0 * sex_shift, 350.0), 2600.0, 6500.0)
                ),
                "island": str(island),
                "sex": str(sex[idx]),
                "species": str(sp),
            }
        )

    return pd.DataFrame(rows)


def generate_bike_sharing(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-01", "2012-12-31", freq="D")
    n_rows = len(dates)

    month = dates.month.to_numpy(dtype=int)
    weekday = dates.weekday.to_numpy()
    season_num = np.select(
        [
            np.isin(month, [12, 1, 2]),
            np.isin(month, [3, 4, 5]),
            np.isin(month, [6, 7, 8]),
            np.isin(month, [9, 10, 11]),
        ],
        [1, 2, 3, 4],
    )
    season = np.select(
        [season_num == 1, season_num == 2, season_num == 3, season_num == 4],
        ["winter", "spring", "summer", "fall"],
        default="unknown",
    )

    base_temp = 0.5 + 0.35 * np.sin((month - 1.0) / 12.0 * 2.0 * np.pi)
    temp = np.clip(base_temp + rng.normal(0.0, 0.08, size=n_rows), 0.0, 1.0)
    atemp = np.clip(temp + rng.normal(0.0, 0.04, size=n_rows), 0.0, 1.0)
    hum = np.clip(
        0.62
        + 0.12 * np.sin((month + 1.0) / 12.0 * 2.0 * np.pi)
        + rng.normal(0.0, 0.08, size=n_rows),
        0.2,
        1.0,
    )
    windspeed = np.clip(0.19 + rng.normal(0.0, 0.05, size=n_rows), 0.02, 0.6)

    weather_score = 0.35 + 0.45 * hum + 0.2 * windspeed
    weathersit_num = np.digitize(weather_score, [0.45, 0.62, 0.78]) + 1
    weathersit = np.select(
        [weathersit_num == 1, weathersit_num == 2, weathersit_num == 3, weathersit_num == 4],
        ["clear", "mist", "light_rain", "heavy_rain"],
        default="unknown",
    )

    holiday = (rng.random(n_rows) < 0.025).astype(int)
    workingday = ((weekday < 5) & (holiday == 0)).astype(int)

    trend = np.linspace(0.0, 220.0, n_rows)
    cnt = (
        550.0
        + trend
        + 1300.0 * temp
        - 520.0 * (weathersit_num - 1)
        - 280.0 * hum
        + 190.0 * workingday
        + rng.normal(0.0, 110.0, size=n_rows)
    )
    cnt = np.clip(cnt, 50.0, None).round().astype(int)

    out = pd.DataFrame(
        {
            "dteday": dates.strftime("%Y-%m-%d"),
            "season": season,
            "yr": (dates.year - dates.year.min()).astype(int),
            "mnth": month,
            "holiday": holiday,
            "workingday": workingday,
            "weathersit": weathersit,
            "temp": temp.round(4),
            "atemp": atemp.round(4),
            "hum": hum.round(4),
            "windspeed": windspeed.round(4),
            "cnt": cnt,
        }
    )
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_csv(url: str) -> pd.DataFrame:
    try:
        with urlrequest.urlopen(url, timeout=30) as response:
            return pd.read_csv(response)
    except urlerror.URLError as exc:
        raise RuntimeError(f"Failed to download dataset from {url}") from exc


def _normalize_lalonde(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if "race" in frame.columns:
        race = frame["race"].astype(str).str.lower()
        frame["black"] = (race == "black").astype(int)
        frame["hispan"] = (race == "hispan").astype(int)

    for col in ["treat", "re78", "age", "educ", "re74", "re75"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    normalized = pd.DataFrame(
        {
            "treatment": frame["treat"].astype("Int64"),
            "outcome": frame["re78"],
            "x1": frame["re74"],
            "x2": frame["re75"],
            "age": frame["age"],
            "educ": frame["educ"],
            "black": frame.get("black", 0),
            "hispan": frame.get("hispan", 0),
        }
    ).dropna(subset=["treatment", "outcome", "x1", "x2"])
    normalized["treatment"] = normalized["treatment"].astype(int)
    return normalized.reset_index(drop=True)


def _normalize_cps_panel(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()

    def _pick(candidates: list[str]) -> str | None:
        for col in candidates:
            if col in frame.columns:
                return col
        return None

    unit_col = _pick(["unit_id", "id", "unitid", "i", "person_id"])
    time_col = _pick(["time", "year", "period", "t"])
    treatment_col = _pick(["treatment", "treated", "treat", "D"])
    target_col = _pick(["target", "outcome", "re", "re78", "y"])
    age_col = _pick(["age", "age_years", "x1"])
    skill_col = _pick(["skill", "educ", "education", "x2"])

    missing_core = [
        name
        for name, col in [
            ("unit_id", unit_col),
            ("time", time_col),
            ("treatment", treatment_col),
            ("target", target_col),
        ]
        if col is None
    ]
    if missing_core:
        raise ValueError(f"CPS panel normalization missing required columns: {missing_core}")

    normalized = pd.DataFrame(
        {
            "__row_id": frame.index,
            "unit_id": frame[unit_col].astype(str),
            "time": pd.to_numeric(frame[time_col], errors="coerce"),
            "treatment": pd.to_numeric(frame[treatment_col], errors="coerce"),
            "target": pd.to_numeric(frame[target_col], errors="coerce"),
            "age": (
                pd.to_numeric(frame[age_col], errors="coerce")
                if age_col is not None
                else pd.Series(0.0, index=frame.index, dtype=float)
            ),
            "skill": (
                pd.to_numeric(frame[skill_col], errors="coerce")
                if skill_col is not None
                else pd.Series(0.0, index=frame.index, dtype=float)
            ),
        }
    )
    normalized = normalized.dropna(subset=["time", "treatment", "target", "age", "skill"])
    normalized["time"] = normalized["time"].astype(int)
    normalized["treatment"] = normalized["treatment"].astype(int)
    normalized = normalized[normalized["treatment"].isin([0, 1])].reset_index(drop=True)
    if normalized.empty:
        raise ValueError("CPS panel normalization produced empty dataset.")

    if "post" in frame.columns:
        post_series = pd.to_numeric(frame["post"], errors="coerce")
        normalized["post"] = post_series.reindex(normalized["__row_id"]).astype("Int64")
        normalized["post"] = normalized["post"].fillna(0).astype(int)
    else:
        min_time = int(normalized["time"].min())
        normalized["post"] = (normalized["time"] > min_time).astype(int)

    return normalized.loc[
        :, ["unit_id", "time", "post", "treatment", "age", "skill", "target"]
    ]


def _write_sources_manifest(out_dir: Path, sources: dict[str, dict[str, str]]) -> None:
    payload = {"datasets": sources}
    for path in (out_dir / SOURCES_MANIFEST_CANONICAL.name, out_dir / SOURCES_MANIFEST_COMPAT.name):
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def fetch_snapshot_datasets(out_dir: Path) -> None:
    retrieved_at = datetime.now(timezone.utc).isoformat()

    lalonde_df = _normalize_lalonde(_download_csv(LALONDE_URL))
    lalonde_path = out_dir / "lalonde.csv"
    lalonde_df.to_csv(lalonde_path, index=False)

    cps_panel_df = _normalize_cps_panel(_download_csv(CPS_PANEL_URL))
    cps_panel_path = out_dir / "cps_panel.csv"
    cps_panel_df.to_csv(cps_panel_path, index=False)

    _write_sources_manifest(
        out_dir,
        {
            "lalonde": {
                "source_url": LALONDE_URL,
                "retrieved_at_utc": retrieved_at,
                "sha256": _sha256_file(lalonde_path),
                "license_note": "Refer to upstream dataset license and terms.",
                "transform_version": SNAPSHOT_TRANSFORM_VERSION,
            },
            "cps_panel": {
                "source_url": CPS_PANEL_URL,
                "retrieved_at_utc": retrieved_at,
                "sha256": _sha256_file(cps_panel_path),
                "license_note": "Refer to upstream dataset license and terms.",
                "transform_version": SNAPSHOT_TRANSFORM_VERSION,
            },
        },
    )


def _validate_contract(
    path: Path,
    required_cols: list[str],
    min_rows: int,
    target_col: str,
    target_type: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    frame = pd.read_csv(path)
    if len(frame) < min_rows:
        raise ValueError(f"{path} expected >= {min_rows} rows, got {len(frame)}")
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if target_type == "numeric":
        if not pd.api.types.is_numeric_dtype(frame[target_col]):
            raise TypeError(f"{path} target column '{target_col}' must be numeric")
        return
    if target_type == "categorical":
        if pd.api.types.is_numeric_dtype(frame[target_col]):
            raise TypeError(f"{path} target column '{target_col}' must be categorical")
        return
    raise ValueError(f"Unsupported target_type: {target_type}")


def validate_outputs(out_dir: Path) -> None:
    _validate_contract(
        out_dir / "ames_housing.csv",
        ["LotArea", "Neighborhood", "HouseStyle", "BldgType", "SalePrice"],
        min_rows=1000,
        target_col="SalePrice",
        target_type="numeric",
    )
    titanic = out_dir / "titanic.csv"
    _validate_contract(
        titanic,
        ["Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"],
        min_rows=500,
        target_col="Survived",
        target_type="numeric",
    )
    titanic_df = pd.read_csv(titanic)
    valid_binary = set(titanic_df["Survived"].dropna().astype(int).unique().tolist()) <= {0, 1}
    if not valid_binary:
        raise ValueError(f"{titanic} Survived must be binary 0/1")

    _validate_contract(
        out_dir / "penguins.csv",
        ["bill_length_mm", "flipper_length_mm", "island", "sex", "species"],
        min_rows=150,
        target_col="species",
        target_type="categorical",
    )

    bike = out_dir / "bike_sharing.csv"
    _validate_contract(
        bike,
        ["dteday", "season", "weathersit", "temp", "hum", "cnt"],
        min_rows=365,
        target_col="cnt",
        target_type="numeric",
    )
    bike_df = pd.read_csv(bike)
    try:
        pd.to_datetime(bike_df["dteday"])
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"{bike} dteday is not parseable as datetime") from exc

    lalonde = out_dir / "lalonde.csv"
    _validate_contract(
        lalonde,
        ["treatment", "outcome", "x1", "x2"],
        min_rows=100,
        target_col="outcome",
        target_type="numeric",
    )
    lalonde_df = pd.read_csv(lalonde)
    if not set(lalonde_df["treatment"].dropna().astype(int).unique().tolist()) <= {0, 1}:
        raise ValueError(f"{lalonde} treatment must be binary 0/1")

    cps = out_dir / "cps_panel.csv"
    _validate_contract(
        cps,
        ["unit_id", "time", "post", "treatment", "age", "skill", "target"],
        min_rows=100,
        target_col="target",
        target_type="numeric",
    )
    cps_df = pd.read_csv(cps)
    if not set(cps_df["treatment"].dropna().astype(int).unique().tolist()) <= {0, 1}:
        raise ValueError(f"{cps} treatment must be binary 0/1")
    if not set(cps_df["post"].dropna().astype(int).unique().tolist()) <= {0, 1}:
        raise ValueError(f"{cps} post must be binary 0/1")
    if cps_df["time"].nunique() < 2:
        raise ValueError(f"{cps} requires at least 2 time periods")

    sources_path = out_dir / SOURCES_MANIFEST_CANONICAL.name
    if not sources_path.exists():
        compat_path = out_dir / SOURCES_MANIFEST_COMPAT.name
        if compat_path.exists():
            sources_path = compat_path
        else:
            raise FileNotFoundError(
                "Missing snapshot source manifest: expected "
                f"{out_dir / SOURCES_MANIFEST_CANONICAL.name}"
            )
    payload = json.loads(sources_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", {})
    for key in ["lalonde", "cps_panel"]:
        if key not in datasets:
            raise ValueError(f"{sources_path} missing datasets.{key}")
        for field in [
            "source_url",
            "retrieved_at_utc",
            "sha256",
            "license_note",
            "transform_version",
        ]:
            if not datasets[key].get(field):
                raise ValueError(f"{sources_path} missing datasets.{key}.{field}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output data directory (default: data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate required files/columns/types without regenerating CSV files.",
    )
    parser.add_argument(
        "--fetch-snapshots",
        action="store_true",
        help=(
            "Fetch and normalize public snapshot datasets into "
            "data/lalonde.csv and data/cps_panel.csv."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fetch_snapshots:
        fetch_snapshot_datasets(out_dir)

    if not args.check_only:
        generate_ames_housing(args.seed).to_csv(out_dir / "ames_housing.csv", index=False)
        generate_titanic(args.seed + 1).to_csv(out_dir / "titanic.csv", index=False)
        generate_penguins(args.seed + 2).to_csv(out_dir / "penguins.csv", index=False)
        generate_bike_sharing(args.seed + 3).to_csv(out_dir / "bike_sharing.csv", index=False)

    validate_outputs(out_dir)
    print(f"Quick-reference datasets are ready under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
