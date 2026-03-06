import pandas as pd

from main import run_pipeline


def test_pipeline_integration_with_mocks(monkeypatch, tmp_path):
    market = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="D", tz="America/Sao_Paulo"),
            "open": [100 + i for i in range(20)],
            "high": [101 + i for i in range(20)],
            "low": [99 + i for i in range(20)],
            "close": [100 + i for i in range(20)],
            "volume": [1000] * 20,
        }
    )

    macro = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="D", tz="America/Sao_Paulo"),
            "selic": [11.75] * 20,
            "DGS10": [4.0] * 20,
            "^GSPC_close": [4800.0] * 20,
        }
    )

    monkeypatch.setattr("main.MT5Connector.fetch_ohlcv", lambda self, **kwargs: market)
    monkeypatch.setattr("main.MacroDataFetcher.build_macro_frame", lambda self, start, end: macro)

    output_file = tmp_path / "dv_resultados.csv"
    dv = run_pipeline(
        symbol="WIN$N",
        start="2024-01-01",
        end="2024-01-30",
        use_ml=True,
        atr_period=14,
        multipliers=[1, 2, 3, 4],
        output_path=str(output_file),
    )

    expected_cols = {
        "calculation_date",
        "target_date",
        "ref_price",
        "atr_pred",
        "dv_plus_1",
        "dv_plus_2",
        "dv_plus_3",
        "dv_plus_4",
        "dv_minus_1",
        "dv_minus_2",
        "dv_minus_3",
        "dv_minus_4",
    }
    assert expected_cols.issubset(set(dv.columns))
    assert output_file.exists()
    assert output_file.with_suffix(".xlsx").exists()
