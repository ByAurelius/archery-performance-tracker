"""Unit tests for analytics.py on small hand-made datasets.

Run with:  python -m pytest
"""
import math

import numpy as np
import pandas as pd
import pytest

import analytics as an


def make_sessions(avgs, start='2026-01-01', **extra):
    dates = pd.date_range(start, periods=len(avgs), freq='D')
    df = pd.DataFrame({'session_id': range(1, len(avgs) + 1), 'date': dates, 'avg': avgs,
                       'arrows_shot': 100, 'total_score': [round(a * 100) for a in avgs]})
    for col, values in extra.items(): df[col] = values
    return df


def make_arrows(points, date='2026-01-01', per_end=6):
    """points: list of (x_mm, y_mm, score) for a single session, in shooting order."""
    rows = [dict(x_mm=x, y_mm=y, score=s, end_no=i // per_end + 1, arrow_no=i % per_end + 1,
                 arrows_per_end=per_end, session_id=1, date=pd.Timestamp(date))
            for i, (x, y, s) in enumerate(points)]
    return pd.DataFrame(rows)


def test_parse_number():
    assert an.parse_number('12.5 km/h') == 12.5
    assert an.parse_number('-3.1 °C') == -3.1
    assert math.isnan(an.parse_number('-'))
    assert math.isnan(an.parse_number(None))


def test_group_stats_mpi_and_r50():
    # Four arrows in a square around (100, 0): MPI is (100, 0), every arrow is 10*sqrt(2) mm from it
    g = an.group_stats([90, 110, 90, 110], [-10, -10, 10, 10])
    assert (g['mpi_x'], g['mpi_y'], g['mpi_dist']) == (100.0, 0.0, 100.0)
    assert g['r50'] == pytest.approx(10 * math.sqrt(2), abs=0.1)


def test_linear_trend_recovers_slope():
    s = make_sessions([7 + 0.01 * d for d in range(90)])
    t = an.linear_trend(s)
    assert t['slope_per_30'] == pytest.approx(0.3, abs=1e-3)
    assert t['r2'] == pytest.approx(1.0)


def test_linear_trend_needs_three_sessions():
    assert an.linear_trend(make_sessions([7, 8])) is None


def test_filter_period_counts_back_from_latest_session():
    s = make_sessions([7.0] * 100)
    a = make_arrows([(0, 0, 10)], date='2026-01-01')
    fs, fa = an.filter_period(s, a, 30)
    assert len(fs) == 30 and fs['date'].min() == pd.Timestamp('2026-03-12')
    assert fa.empty  # the only arrow is from day 1


def test_summary_change_30_compares_last_two_months():
    s = make_sessions([7.0] * 30 + [8.0] * 30)
    a = make_arrows([(0, 0, 10), (0, 0, 0)])
    m = an.summary(s, a)
    assert m['change_30'] == 1.0
    assert m['pct_10'] == 50.0 and m['pct_miss'] == 50.0


def test_monthly_boxplot_flags_outliers():
    s = make_sessions([8.0] * 10 + [8.1] * 10 + [2.0])
    box = an.monthly_boxplot(s)
    assert len(box) == 1
    assert box[0]['median'] == 8.0 and box[0]['outliers'] == [2.0]
    assert box[0]['whisker_low'] == 8.0


def test_in_session_fatigue_halves():
    # 12 arrows: first six score 10, last six score 8
    a = make_arrows([(0, 0, 10)] * 6 + [(0, 0, 8)] * 6)
    f = an.in_session_fatigue(a)
    assert f['first_half'] == 10 and f['second_half'] == 8 and f['drop'] == 2
    assert f['end_avg'] == [10, 8]


def test_correlation_sign_and_detrending():
    rng = np.random.default_rng(0)
    wind = rng.uniform(0, 20, 200)
    avg = 8 - 0.05 * wind + np.linspace(0, 1, 200)  # wind hurts, plus a learning trend
    s = make_sessions(list(avg), wind=wind, temp=np.linspace(0, 20, 200),
                      fatigue_level=5, focus_level=5)
    c = {f['key']: f for f in an.correlations(s)['factors']}
    assert c['wind']['r'] < -0.5 and c['wind']['r_detrended'] < -0.5
    # Temperature only rises with time, so its raw correlation is spurious and disappears once detrended
    assert c['temp']['r'] > 0.5 and abs(c['temp']['r_detrended']) < 0.3
    assert c['fatigue_level']['r'] is None  # constant column -> undefined


def test_plan_effect_groups_completed_and_open():
    s = make_sessions([7.0] * 14 + [8.0] * 14 + [7.0] * 28)
    plans = pd.DataFrame({'date_assigned': pd.to_datetime(['2026-01-15', '2026-02-12']),
                          'is_completed': [True, False]})
    e = an.plan_effect(plans, s)
    assert e['rate'] == 50.0
    assert e['change_completed'] == 1.0 and e['change_open'] == 0.0


def test_score_distribution_shares_sum_to_100():
    a = make_arrows([(0, 0, 10), (0, 0, 9), (0, 0, 9), (0, 0, 0)])
    d = an.score_distribution(a)
    assert sum(d['all']) == pytest.approx(100)
    assert d['all'][0] == 25 and d['all'][9] == 50
