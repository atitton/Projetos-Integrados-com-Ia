import pandas as pd

from technical_indicators import calculate_atr


def test_calculate_atr_known_values():
    df = pd.DataFrame(
        {
            "high": [10, 12, 11, 13],
            "low": [8, 9, 9, 10],
            "close": [9, 11, 10, 12],
        }
    )
    atr = calculate_atr(df, period=3)
    # TR: [2, 3, 2, 3] -> ATR(3): [nan, nan, 7/3, 8/3]
    assert round(atr.iloc[2], 6) == round(7 / 3, 6)
    assert round(atr.iloc[3], 6) == round(8 / 3, 6)
