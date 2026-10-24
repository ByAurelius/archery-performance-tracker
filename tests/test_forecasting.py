"""Unit tests for forecasting.py: windowing, leakage boundaries, baselines and shapes.

Run with:  python -m pytest
"""
import numpy as np
import pandas as pd
import pytest
import torch

import forecasting as fc


def make_df(n, avg=None):
    avg = np.linspace(6, 8, n) if avg is None else np.asarray(avg, float)
    return pd.DataFrame({'date': pd.date_range('2026-01-01', periods=n, freq='D'), 'avg': avg,
                         'score_sd': 1.0 + 0.01 * np.arange(n), 'r50': 100.0, 'fatigue_level': 5.0,
                         'focus_level': np.arange(n) % 10, 'wind': 10.0})


def test_split_targets_do_not_overlap():
    n = 365
    target_days = {s: {o + h for o in fc.split_origins(n, s) for h in range(fc.HORIZON)} for s in ['train', 'val', 'test']}
    assert max(target_days['train']) < fc.TRAIN_END <= min(target_days['val'])
    assert max(target_days['val']) < fc.VAL_END <= min(target_days['test'])
    assert max(target_days['test']) == n - 1


def test_windows_use_only_past_inputs():
    df = make_df(200)
    scaler = fc.Scaler(df)
    x, avg = scaler.transform(df), df['avg'].to_numpy()
    X, y, anchors = fc.make_windows(x, avg, scaler.avg_sd, [100])
    assert X.shape == (1, fc.INPUT_DAYS, len(fc.FEATURES)) and y.shape == (1, fc.HORIZON)
    np.testing.assert_allclose(X[0, -1], x[99], rtol=1e-5)  # last input is the day before the origin
    assert anchors[0] == pytest.approx(avg[100 - fc.ANCHOR_DAYS:100].mean())
    np.testing.assert_allclose(anchors[0] + y[0] * scaler.avg_sd, avg[100:100 + fc.HORIZON], rtol=1e-5)


def test_scaler_fits_on_training_rows_only():
    df = make_df(100, avg=[1.0] * 50 + [100.0] * 50)
    scaler = fc.Scaler(df, fit_rows=50)
    assert scaler.mean[0] == 1.0 and scaler.avg_sd == 1.0  # constant column -> SD replaced by 1


def test_baselines():
    avg = np.arange(100, dtype=float)
    b = fc.baseline_forecasts(avg, 100)
    assert b['Persistence (last 7)'][0] == pytest.approx(np.mean(range(93, 100)))
    assert b['Moving average (last 30)'][0] == pytest.approx(np.mean(range(70, 100)))
    np.testing.assert_allclose(b['Linear trend (last 90)'], np.arange(100, 100 + fc.HORIZON), atol=1e-6)


def test_error_metrics():
    m = fc.error_metrics([[1.0, 3.0]], [[2.0, 2.0]])
    assert m['mae'] == 1.0 and m['rmse'] == 1.0 and m['month_error'] == 0.0


def test_model_output_shape_and_forecast_rows():
    model = fc.LSTMForecaster()
    assert model(torch.zeros(3, fc.INPUT_DAYS, len(fc.FEATURES))).shape == (3, fc.HORIZON)
    rows = fc.forecast(model, make_df(80))
    assert len(rows) == fc.HORIZON
    assert rows[0]['date'] == pd.Timestamp('2026-03-22').date()  # the day after the last session
    assert all(r['lower'] <= r['upper'] for r in rows)


def test_forecast_needs_enough_sessions():
    assert fc.forecast(fc.LSTMForecaster(), make_df(fc.INPUT_DAYS - 1)) is None
