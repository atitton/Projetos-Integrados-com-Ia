import pandas as pd

from ml_model import ATRModel


class FakeRegressor:
    def fit(self, x, y):
        self.mean_ = float(y.mean())

    def predict(self, x):
        return [self.mean_] * len(x)


def test_ml_model_predict_with_fake_regressor():
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 4.0, 6.0, 8.0],
            "atr_target": [10.0, 12.0, 14.0, 16.0],
        }
    )

    model = ATRModel()
    model.model = FakeRegressor()
    model.train(df, feature_cols=["f1", "f2"], target_col="atr_target")
    pred = model.predict(df[["f1", "f2"]])

    assert (pred == 13.0).all()
