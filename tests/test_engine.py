import numpy as np
import pandas as pd
import pytest

from shrimp_forecast.engine import TrainConfig, backtest, calibrate, forecast, predict_panel, splits, train
from shrimp_forecast.features import next_dates, row_features, training_matrix


def test_origin_features_ignore_future_targets_and_delayed_releases(panel):
    selected = panel[panel.series_id.eq(panel.series_id.iloc[0])].copy()
    date = pd.Timestamp("2024-03-04")
    before, _ = row_features(selected, date, "farmgate")
    selected.loc[selected.date.ge(date), "price_usd_kg"] *= 1000
    after, _ = row_features(selected, date, "farmgate")
    assert before == after
    last = selected.date.eq(date - pd.Timedelta(weeks=1))
    selected.loc[last, "available_at"] = date + pd.Timedelta(days=1)
    _, anchor = row_features(selected, date, "farmgate")
    assert anchor == selected[selected.date.eq(date - pd.Timedelta(weeks=2))].price_usd_kg.iloc[0]


def test_training_features_have_no_target_column(panel):
    features, targets = training_matrix(panel, "farmgate")
    assert "price_usd_kg" not in features.columns
    assert len(features) == len(targets)
    assert np.isfinite(targets).all()


def test_missing_period_is_a_calendar_gap(panel):
    selected = panel[panel.series_id.eq(panel.series_id.iloc[0])].copy()
    selected = selected[selected.date.ne(pd.Timestamp("2024-03-04"))]
    features, _ = row_features(selected, pd.Timestamp("2024-03-11"), "farmgate")
    assert features["gap"] == 1
    assert np.isnan(features["lag_return"])


def test_partition_target_dates_are_disjoint(panel):
    config = TrainConfig()
    parts = splits(panel, config)
    targets = {
        key: set(date for origin in origins for date in next_dates(origin, "farmgate", config.max_horizon))
        for key, origins in parts.items()
    }
    assert max(targets["validation"]) < min(targets["calibration"])
    assert max(targets["calibration"]) < min(targets["test"])


def test_recursive_backtest_never_uses_intermediate_actual_targets(panel):
    origin = pd.Timestamp("2024-03-04")
    predictions = predict_panel(panel, "farmgate", "latest_price", None, origin, 4)
    mutated = panel.copy()
    mutated.loc[mutated.date.gt(origin), "price_usd_kg"] *= 10
    repeated = predict_panel(mutated, "farmgate", "latest_price", None, origin, 4)
    pd.testing.assert_frame_equal(predictions, repeated)
    assert predictions.groupby("series_id").prediction.nunique().eq(1).all()


def test_backtest_metrics_include_all_recursive_horizons(panel):
    config = TrainConfig()
    rows = backtest(panel, "farmgate", "latest_price", config, [pd.Timestamp("2024-03-04")])
    assert set(rows.horizon) == {1, 2, 3, 4}
    assert len(rows) == 12 * 4


def test_calibration_does_not_invent_small_sample_bounds():
    rows = pd.DataFrame({"prediction": [4.0, 4.0], "price_usd_kg": [4.1, 4.2], "horizon": [1, 1]})
    result = calibrate(rows, TrainConfig())
    assert result["1"]["log_radius"] is None


def test_forecast_uses_held_out_calibration_and_limits_horizon(bundle):
    result = forecast(bundle, "in:vannamei:40", 4)
    assert len(result["predictions"]) == 4
    assert result["interval"]["guaranteed_coverage"] is False
    for row in result["predictions"]:
        assert 0 <= row["lower"] <= row["prediction"] <= row["upper"]
    with pytest.raises(ValueError, match="calibrated"):
        forecast(bundle, "in:vannamei:40", 5)
    with pytest.raises(KeyError):
        forecast(bundle, "missing", 1)


def test_final_test_cannot_change_model_selection_or_calibration(panel, bundle):
    modified = panel.copy()
    test_origin = pd.Timestamp(bundle["report"]["partitions"]["test"][0])
    modified.loc[modified.date.gt(test_origin), "price_usd_kg"] *= 3
    rerun = train(modified, "farmgate", bundle["config"])
    assert bundle["name"] == rerun["name"]
    assert bundle["report"]["validation"] == rerun["report"]["validation"]
    assert bundle["calibration"] == rerun["calibration"]
    assert bundle["report"]["test"]["mae"] != rerun["report"]["test"]["mae"]


def test_short_history_fails_explicitly(panel):
    with pytest.raises(ValueError, match="independent"):
        train(panel[panel.date.lt("2024-03-01")], "farmgate")


def test_off_cycle_release_is_available_at_live_origin(panel):
    selected = panel[panel.series_id.eq(panel.series_id.iloc[0])].copy()
    origin = selected.date.max() + pd.Timedelta(days=3)
    selected.loc[selected.date.eq(selected.date.max()), "available_at"] = origin
    result = predict_panel(selected, "farmgate", "latest_price", None, origin, 1)
    assert result.prediction.iloc[0] == selected.price_usd_kg.iloc[-1]
    assert result.date.iloc[0] > origin
