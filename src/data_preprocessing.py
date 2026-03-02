import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

@dataclass
class Preprocessor:
    """
    From-scratch preprocessor for the Life Expectancy task.

    - Normalizes column names.
    - Applies EDA rules (zero->NaN for specific cols, ffill/bfill by country,
      status/year medians, winsorize BMI, log1p on skewed counts).
    - Encodes status (drop-first one-hot), drops country & year from features.
    - Scales numerics with stored mean/std (no sklearn).
    - Returns: X (with bias column), y.
    """
    target_col: str = "life_expectancy"
    country_col: str = "country"
    year_col: str = "year"
    status_col: str = "status"

    # columns to log-transform (counts)
    log_cols: List[str] = field(default_factory=lambda: ["measles", "infant_deaths", "under-five_deaths"])
    # columns to drop from features
    drop_cols: List[str] = field(default_factory=lambda: ["country", "year"])

    # learned params
    num_means_: Dict[str, float] = field(default_factory=dict)
    num_stds_: Dict[str, float] = field(default_factory=dict)
    status_values_: List[str] = field(default_factory=list)   # order fixed at fit
    feature_names_: List[str] = field(default_factory=list)

    # group medians learned at fit time (prevent leakage)
    med_global_: Dict[str, float] = field(default_factory=dict)
    med_status_: Dict[str, Dict[str, float]] = field(default_factory=dict)
    med_status_year_: Dict[str, Dict[Tuple[str, Any], float]] = field(default_factory=dict)

    # ---------- helpers ----------
    def _normalize_names(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = (out.columns.str.strip()
                                  .str.lower()
                                  .str.replace(" ", "_"))
        # canonicalize a few known variants
        ren = {}
        if "thinness__1-19_years" in out.columns and "thinness_1-19_years" not in out.columns:
            ren["thinness__1-19_years"] = "thinness_1-19_years"
        if "under_five_deaths" in out.columns and "under-five_deaths" not in out.columns:
            ren["under_five_deaths"] = "under-five_deaths"
        if ren:
            out = out.rename(columns=ren)
        return out

    def _to_numeric_safely(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            if c not in (self.status_col, self.country_col):
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def _sort_for_time(self, df: pd.DataFrame) -> pd.DataFrame:
        if {self.country_col, self.year_col}.issubset(df.columns):
            return df.sort_values([self.country_col, self.year_col], kind="mergesort").reset_index(drop=True)
        return df

    def _ffill_bfill_by_country(self, df: pd.DataFrame, col: str) -> pd.Series:
        if self.country_col in df.columns:
            return (df.groupby(self.country_col, group_keys=False)[col]
                      .apply(lambda g: g.ffill().bfill()))
        return df[col]

    def _zero_to_nan(self, df: pd.DataFrame, cols: List[str]) -> None:
        for c in cols:
            if c in df.columns:
                df.loc[df[c] == 0, c] = np.nan

    def _learn_group_medians(self, df: pd.DataFrame, cols: List[str]) -> None:
        for c in cols:
            # global
            self.med_global_[c] = float(df[c].median(skipna=True)) if c in df.columns else np.nan
            # by status
            if self.status_col in df.columns and c in df.columns:
                self.med_status_[c] = df.groupby(self.status_col)[c].median().to_dict()
            else:
                self.med_status_[c] = {}
            # by (status, year)
            if {self.status_col, self.year_col}.issubset(df.columns) and c in df.columns:
                tmp = df.groupby([self.status_col, self.year_col])[c].median()
                self.med_status_year_[c] = { (s,y): float(v) for (s,y), v in tmp.items() }
            else:
                self.med_status_year_[c] = {}

    def _impute_series(self, df: pd.DataFrame, col: str) -> pd.Series:
        s = df[col].copy()
        # status-year median
        if {self.status_col, self.year_col}.issubset(df.columns) and self.med_status_year_.get(col):
            key = list(zip(df[self.status_col], df[self.year_col]))
            vals = [self.med_status_year_[col].get(k, np.nan) for k in key]
            s = s.fillna(pd.Series(vals, index=s.index))
        # status-only median
        if self.status_col in df.columns and self.med_status_.get(col):
            vals = df[self.status_col].map(self.med_status_[col]).astype(float)
            s = s.fillna(vals)
        # global median
        if col in self.med_global_:
            s = s.fillna(self.med_global_[col])
        return s

    def _one_hot_status(self, s: pd.Series) -> np.ndarray:
        if not len(self.status_values_):
            return np.zeros((len(s), 0), dtype=float)
        first = self.status_values_[0]  # drop-first baseline
        out = np.zeros((len(s), len(self.status_values_) - 1), dtype=float)
        for i, cat in enumerate(self.status_values_[1:]):
            out[:, i] = (s.values == cat).astype(float)
        return out


    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        df = self._normalize_names(df)
        df = self._to_numeric_safely(df)
        df = self._sort_for_time(df)

        # zero -> NaN 
        self._zero_to_nan(df, [
            "gdp", "population", "adult_mortality", "bmi",
            "percentage_expenditure", "income_composition_of_resources",
            # NOTE: do NOT zero->NaN for measles/infant/under-five; zeros can be real
        ])

        # vaccination %: smooth over time, then status median fallback
        for c in ["polio", "diphtheria", "hepatitis_b"]:
            if c in df.columns:
                df[c] = self._ffill_bfill_by_country(df, c)

        # percentage_expenditure: temporal continuity helps
        if "percentage_expenditure" in df.columns:
            df["percentage_expenditure"] = self._ffill_bfill_by_country(df, "percentage_expenditure")

        # GDP/population: continuity too
        for c in ["gdp", "population"]:
            if c in df.columns:
                df[c] = self._ffill_bfill_by_country(df, c)

        # schooling: isolated zeros -> NaN (after sorting)
        if "schooling" in df.columns and self.country_col in df.columns:
            prev = df.groupby(self.country_col)["schooling"].shift(1)
            nxt  = df.groupby(self.country_col)["schooling"].shift(-1)
            mask = (df["schooling"] == 0) & ((prev > 0) | (nxt > 0))
            df.loc[mask, "schooling"] = np.nan

        # learn medians for imputation (prevents leakage)
        cols_to_impute = [c for c in [
            "polio","diphtheria","hepatitis_b",      # vax
            "measles",                               # spikes but we only median-impute missing
            "gdp","population",
            "adult_mortality","bmi",
            "total_expenditure",
            "income_composition_of_resources",
            "schooling","percentage_expenditure",
            "thinness_1-19_years","thinness_5-9_years",
            "alcohol"
        ] if c in df.columns]
        self._learn_group_medians(df, cols_to_impute)

        # impute using learned medians
        for c in cols_to_impute:
            df[c] = self._impute_series(df, c)

        # winsorize BMI
        if "bmi" in df.columns:
            df["bmi"] = df["bmi"].clip(lower=12, upper=60)

        # drop rows missing target
        assert self.target_col in df.columns, f"Target '{self.target_col}' missing."
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)

        # log1p on skewed counts (keep zeros valid)
        for c in self.log_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c])

        # status categories (fixed order)
        if self.status_col in df.columns:
            self.status_values_ = pd.Series(df[self.status_col].astype(str)).fillna("MISSING").unique().tolist()
        else:
            self.status_values_ = []

        # choose feature columns for scaling stats
        X_df = df.drop(columns=[self.target_col], errors="ignore")
        for c in self.drop_cols:
            X_df = X_df.drop(columns=[c], errors="ignore")

        num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()

        # learn scaling params
        self.num_means_.clear(); self.num_stds_.clear()
        for c in num_cols:
            m = float(np.nanmean(X_df[c]))
            s = float(np.nanstd(X_df[c]))
            self.num_means_[c] = m
            self.num_stds_[c] = s if s > 0 else 1.0

        # freeze feature order
        self.feature_names_ = ["bias"] + num_cols + (
            [f"{self.status_col}_{v}" for v in self.status_values_[1:]] if self.status_values_ else []
        )

        return self

    def _scale_numeric(self, df_num: pd.DataFrame, num_cols: List[str]) -> np.ndarray:
        mats = []
        for c in num_cols:
            col = pd.to_numeric(df_num[c], errors="coerce").to_numpy()
            col = (col - self.num_means_[c]) / self.num_stds_[c]
            mats.append(col.reshape(-1,1))
        return np.hstack(mats) if mats else np.zeros((len(df_num), 0))

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = self._normalize_names(df)
        df = self._to_numeric_safely(df)
        df = self._sort_for_time(df)

        # apply same rules deterministically
        self._zero_to_nan(df, [
            "gdp","population","adult_mortality","bmi",
            "percentage_expenditure","income_composition_of_resources"
        ])
        for c in ["polio","diphtheria","hepatitis_b"]:
            if c in df.columns:
                df[c] = self._ffill_bfill_by_country(df, c)
        if "percentage_expenditure" in df.columns:
            df["percentage_expenditure"] = self._ffill_bfill_by_country(df, "percentage_expenditure")
        for c in ["gdp", "population"]:
            if c in df.columns:
                df[c] = self._ffill_bfill_by_country(df, c)
        if "schooling" in df.columns and self.country_col in df.columns:
            prev = df.groupby(self.country_col)["schooling"].shift(1)
            nxt  = df.groupby(self.country_col)["schooling"].shift(-1)
            mask = (df["schooling"] == 0) & ((prev > 0) | (nxt > 0))
            df.loc[mask, "schooling"] = np.nan

        # impute using learned medians
        for c in self.med_global_.keys():
            if c in df.columns:
                df[c] = self._impute_series(df, c)

        # winsorize BMI again
        if "bmi" in df.columns:
            df["bmi"] = df["bmi"].clip(lower=12, upper=60)

        # log1p again on same columns
        for c in self.log_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c])

        # ensure target present
        assert self.target_col in df.columns, f"Target '{self.target_col}' missing."
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)

        y = df[self.target_col].astype(float).to_numpy()

        # build X using frozen order
        X_df = df.drop(columns=[self.target_col], errors="ignore")
        for c in self.drop_cols:
            X_df = X_df.drop(columns=[c], errors="ignore")

        num_cols = [c for c in X_df.select_dtypes(include=[np.number]).columns if c in self.num_means_]
        num_scaled = self._scale_numeric(X_df[num_cols], num_cols)

        if self.status_values_ and self.status_col in X_df.columns:
            s = X_df[self.status_col].astype(str).fillna("MISSING")
            status_mat = np.zeros((len(s), len(self.status_values_) - 1), dtype=float)
            for i, cat in enumerate(self.status_values_[1:]):
                status_mat[:, i] = (s.values == cat).astype(float)
            # drop status column after encoding
            # (we don't reorder numeric columns; we join as [bias | nums | status_onehot])
        else:
            status_mat = np.zeros((len(X_df), 0), dtype=float)

        bias = np.ones((len(X_df), 1), dtype=float)
        X = np.hstack([bias, num_scaled, status_mat])
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)
