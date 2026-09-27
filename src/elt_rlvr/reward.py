"""Execution-derived reward; ground truth is never returned to the agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TableScore:
    score: float
    matched_columns: int
    total_columns: int
    row_count_match: bool


def _sorted(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    lookup = {str(c).lower(): c for c in df.columns}
    sort_columns = [lookup[k.lower()] for k in keys if k.lower() in lookup]
    sort_columns.extend(c for c in df.columns if c not in sort_columns)
    if not len(df) or not sort_columns:
        return df.reset_index(drop=True)
    aux = pd.DataFrame(index=df.index)
    for col in sort_columns:
        values = df[col].astype("string").str.strip().str.lower()
        numeric = pd.to_numeric(values, errors="coerce")
        aux[str(col)] = numeric if numeric.notna().sum() == values.notna().sum() else values
    order = aux.sort_values(list(aux.columns), kind="mergesort", na_position="last").index
    return df.loc[order].reset_index(drop=True)


def _equal_series(gold: pd.Series, actual: pd.Series) -> bool:
    g = gold.reset_index(drop=True)
    a = actual.reset_index(drop=True)
    gn, an = pd.to_numeric(g, errors="coerce"), pd.to_numeric(a, errors="coerce")
    if gn.notna().sum() == g.notna().sum() and an.notna().sum() == a.notna().sum():
        return bool(np.isclose(gn.to_numpy(float), an.to_numpy(float), rtol=1e-2, atol=1e-9, equal_nan=True).all())
    gs = g.astype("string").str.strip().str.casefold()
    av = a.astype("string").str.strip().str.casefold()
    return bool(((gs == av) | (gs.isna() & av.isna())).fillna(False).all())


def compare_tables(gold: pd.DataFrame, actual: pd.DataFrame, keys: list[str] | None = None) -> TableScore:
    """Columnwise comparator aligned with ELT-Bench's stage-2 tolerance."""
    if len(gold) != len(actual):
        return TableScore(0.0, 0, len(gold.columns), False)
    gold = _sorted(gold, keys or [])
    actual = _sorted(actual, keys or [])
    lookup = {str(c).lower(): c for c in actual.columns}
    matched = sum(
        1 for col in gold.columns
        if str(col).lower() in lookup and _equal_series(gold[col], actual[lookup[str(col).lower()]])
    )
    total = len(gold.columns)
    return TableScore(matched / total if total else 0.0, matched, total, True)
