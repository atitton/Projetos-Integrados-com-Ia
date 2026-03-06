from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, List

import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday


class BrazilSimpleHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("Confraternizacao Universal", month=1, day=1),
        Holiday("Tiradentes", month=4, day=21),
        Holiday("Dia do Trabalho", month=5, day=1),
        Holiday("Independencia", month=9, day=7),
        Holiday("Nossa Senhora", month=10, day=12),
        Holiday("Finados", month=11, day=2),
        Holiday("Proclamacao da Republica", month=11, day=15),
        Holiday("Natal", month=12, day=25),
    ]


@dataclass
class DVConfig:
    multipliers: Iterable[int] = (1, 2, 3, 4)
    timezone: str = "America/Sao_Paulo"


def next_business_day(date: pd.Timestamp) -> pd.Timestamp:
    holidays = BrazilSimpleHolidayCalendar().holidays(start=date, end=date + pd.Timedelta(days=10))
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day.normalize() in holidays.normalize():
        next_day += timedelta(days=1)
    return next_day


def build_dv_levels(df: pd.DataFrame, config: DVConfig) -> pd.DataFrame:
    work = df.copy()
    work["calculation_date"] = pd.to_datetime(work["date"])
    work["target_date"] = work["calculation_date"].apply(next_business_day)
    work["ref_price"] = work["close"]

    out_cols: List[str] = ["calculation_date", "target_date", "ref_price", "atr_pred"]

    plus_cols = []
    minus_cols = []
    for m in config.multipliers:
        plus_col = f"dv_plus_{m}"
        minus_col = f"dv_minus_{m}"
        work[plus_col] = work["ref_price"] + work["atr_pred"] * m
        work[minus_col] = work["ref_price"] - work["atr_pred"] * m
        plus_cols.append(plus_col)
        minus_cols.append(minus_col)

    return work[out_cols + plus_cols + minus_cols]
