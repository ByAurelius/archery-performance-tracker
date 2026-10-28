"""Diebold-Mariano test: is the LSTM significantly more accurate than the baselines on the test period?

Usage:  python significance_test.py

The evaluation model is retrained exactly as in train_forecast.py (same seed, same chronological split),
every test window is forecast, and the error of the LSTM is compared window by window with each baseline.
The loss of a window is its mean absolute error (and, separately, its mean squared error) over the 30 days.
Consecutive windows share 29 of their 30 target days, so their loss differences are strongly autocorrelated;
the variance of the mean difference is therefore estimated with a Newey-West (Bartlett) estimator with 29 lags,
separately for each archer and then combined. A negative difference means the LSTM was more accurate.
Results are printed and written to models/significance.json.
"""
import json
import math
import os

import numpy as np
from scipy import stats

import forecasting as F
from app import app, db, Archer, Score

LAG = F.HORIZON - 1  # overlapping windows: a difference can be correlated with the next 29 ones
OUT_PATH = os.path.join(F.MODEL_DIR, 'significance.json')


def long_run_variance(d, lag):
    """Newey-West estimate of the long-run variance of a series (Bartlett weights, never negative)."""
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    v = d @ d / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return v


def dm_test(diffs_per_archer):
    """Mean loss difference over all windows, DM statistic and two-sided p-value (normal approximation).
    Archers are independent, so their variance contributions are added."""
    pooled = np.concatenate(diffs_per_archer)
    n = len(pooled)
    var = sum(len(d) * long_run_variance(d, LAG) for d in diffs_per_archer) / n ** 2
    stat = pooled.mean() / math.sqrt(var) if var > 0 else float('nan')
    return dict(mean_diff=round(float(pooled.mean()), 5), dm=round(float(stat), 3),
                p=round(float(2 * stats.norm.sf(abs(stat))), 4), windows=int(n))


def window_losses(pairs, kind):
    err = np.array([p - a for p, a in pairs])
    return np.abs(err).mean(axis=1) if kind == 'abs' else (err ** 2).mean(axis=1)


def main():
    with app.app_context():
        archers = [dict(name=a.first_name or a.username, username=a.username, df=F.load_features(db.engine, a.id))
                   for a in Archer.query.filter_by(is_coach=False).order_by(Archer.id)
                   if Score.query.filter_by(archer_id=a.id).count() >= F.VAL_END + F.HORIZON]
    print('Archers:', ', '.join(a['name'] for a in archers))
    archers = F.prepare(archers)
    X_tr, y_tr = F.stack_split(archers, 'train')
    X_va, y_va = F.stack_split(archers, 'val')
    model, _, best_epoch = F.train_model(X_tr, y_tr, X_va, y_va)
    print(f'Retrained the evaluation model (best epoch {best_epoch}).')

    errors = [F.test_errors(model, a)[0] for a in archers]  # per archer: {method: [(pred, actual), ...]}
    methods = list(errors[0].keys())
    results = dict(lag=LAG, best_epoch=best_epoch, mae={}, comparisons={})
    for m in methods:
        results['mae'][m] = round(float(np.mean(np.concatenate([window_losses(e[m], 'abs') for e in errors]))), 4)
    for m in methods:
        if m == 'LSTM': continue
        results['comparisons'][m] = {
            kind: dm_test([window_losses(e['LSTM'], kind) - window_losses(e[m], kind) for e in errors])
            for kind in ('abs', 'squared')}

    print(f"\nTest-period MAE (should match evaluation.json): {results['mae']}")
    print(f"\n{'LSTM compared with':<32}{'loss':<9}{'mean diff':>10}{'DM':>8}{'p':>8}{'windows':>9}")
    for m, by_kind in results['comparisons'].items():
        for kind, r in by_kind.items():
            print(f"{m:<32}{kind:<9}{r['mean_diff']:>10.4f}{r['dm']:>8.2f}{r['p']:>8.4f}{r['windows']:>9}")
    with open(OUT_PATH, 'w', encoding='utf-8') as f: json.dump(results, f, indent=1)
    print(f'\nSaved {OUT_PATH}')


if __name__ == '__main__':
    main()
