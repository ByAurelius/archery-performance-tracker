"""Unit tests for insights.py: the plain-language labels must match their thresholds.

Run with:  python -m pytest
"""
import numpy as np
import pandas as pd

import insights as ins


def test_consistency_labels():
    assert ins.consistency(0.12)['label'] == 'Very steady'
    assert ins.consistency(0.25)['label'] == 'Steady'
    assert ins.consistency(0.40)['label'] == 'Variable'
    assert ins.consistency(0.60)['label'] == 'Very variable'


def test_trend_labels():
    assert ins.trend(0.2)['label'] == 'Improving'
    assert ins.trend(-0.2)['label'] == 'Declining'
    assert ins.trend(0.01)['label'] == 'Stable'
    assert ins.trend(None)['label'] == 'Not enough data'


def test_grouping_compares_with_ring():
    g = ins.grouping(170)  # 17 cm radius fits inside the 8 ring (radius 18.3 cm)
    assert g['cm'] == 17.0 and g['ring'] == 8


def test_drift_direction_and_centred():
    assert ins.drift(5, -5)['label'] == 'Well centred'
    d = ins.drift(54, -1)
    assert d['label'] == '5 cm right' and d['tone'] == 'warn' and 'sight' in d['text']
    assert ins.drift(-30, 30)['label'] == '3 cm left and 3 cm high'


def test_fatigue_labels():
    assert ins.fatigue(8.0, 7.95)['label'] == 'No real drop'
    f = ins.fatigue(7.5, 7.1)
    assert f['label'] == 'Clear drop' and f['session_points'] == 20  # 0.4 points x last 50 arrows


def test_effect_strength_needs_significance_and_size():
    assert ins.effect_strength(0.5, -0.5) == 'No clear effect'     # not significant
    assert ins.effect_strength(0.001, 0.03) == 'No clear effect'   # 3 points per 100 arrows
    assert ins.effect_strength(0.001, 0.10) == 'Small effect'
    assert ins.effect_strength(0.001, -0.25) == 'Moderate effect'
    assert ins.effect_strength(0.001, 0.98) == 'Strong effect'


def test_factor_effects_find_the_better_condition():
    rng = np.random.default_rng(1)
    n = 200
    wind = rng.uniform(0, 25, n)
    s = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=n, freq='D'),
                      'avg': 8 - 0.04 * wind + rng.normal(0, 0.05, n), 'wind': wind,
                      'fatigue_level': 5.0, 'focus_level': 5.0, 'temp': 10.0})
    correlations = {'factors': [{'key': 'wind', 'r_detrended': -0.9, 'p_detrended': 1e-6}]}
    e = ins.factor_effects(s, correlations)
    wind_effect = next(x for x in e if x['key'] == 'wind')
    assert wind_effect['strength'] in ('Moderate effect', 'Strong effect')
    assert wind_effect['text'].startswith('On calm days you score')
