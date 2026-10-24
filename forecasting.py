"""30-day performance forecast with an LSTM, plus the baselines it is judged against.

Task: from the last INPUT_DAYS sessions of an archer, predict the session average
(mean arrow score) of each of the next HORIZON sessions in one shot.

Design choices:
  * One model is shared by all archers. Every archer's features are standardised
    with their own mean and SD, and the model predicts the change relative to the
    mean of the last ANCHOR_DAYS sessions, so archers on very different levels can
    be learnt together. Anchor length, hidden size and dropout were chosen on the
    validation set only (tune_forecast.py).
  * Chronological split on the day index of the *targets* (train < 245 <= val < 305
    <= test), with gaps so no target day is shared between splits.
  * Uncertainty from Monte Carlo dropout: dropout stays on at prediction time and
    the spread of repeated predictions gives an 80 % band.
  * Baselines: persistence (last 7 sessions), 30-session mean, 90-session linear
    trend. For simulated archers an "oracle" (the simulator's own expected value)
    shows the lowest error reachable in expectation; on a short test period a model
    can come out marginally ahead of it by chance.
"""
import json
import math
import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch import nn

import analytics

INPUT_DAYS = 60
HORIZON = 30
ANCHOR_DAYS = 30  # chosen on the validation set, see tune_forecast.py
HIDDEN, DROPOUT = 64, 0.2
FEATURES = ['avg', 'score_sd', 'r50', 'fatigue_level', 'focus_level', 'wind']
TRAIN_END, VAL_END = 245, 305
SEED = 42
MC_SAMPLES = 100
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_forecast.pt')
EVAL_PATH = os.path.join(MODEL_DIR, 'evaluation.json')


# --- Data ----------------------------------------------------------------------

def session_features(sessions, arrows):
    """One row per session with the model's input features, sorted by date."""
    a = arrows.copy()
    a['cx'] = a.groupby('session_id')['x_mm'].transform('mean')
    a['cy'] = a.groupby('session_id')['y_mm'].transform('mean')
    a['dist'] = np.hypot(a['x_mm'] - a['cx'], a['y_mm'] - a['cy'])
    per_session = a.groupby('session_id').agg(score_sd=('score', 'std'), r50=('dist', 'median')).reset_index()
    df = sessions.merge(per_session, on='session_id', how='inner')
    df = df.dropna(subset=['avg']).sort_values(['date', 'session_id']).reset_index(drop=True)
    # Manually entered sessions may lack readiness or weather values
    for col, default in [('fatigue_level', 5), ('focus_level', 5), ('wind', 0), ('score_sd', 0), ('r50', 0)]:
        df[col] = df[col].astype(float)
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else default)
    return df[['date'] + FEATURES]


def load_features(engine, archer_id):
    return session_features(analytics.load_sessions(engine, archer_id), analytics.load_arrows(engine, archer_id))


class Scaler:
    """Per-archer standardisation fitted on the first `fit_rows` sessions only."""

    def __init__(self, df, fit_rows=None):
        fit = df[FEATURES].iloc[:fit_rows].to_numpy(float)
        self.mean, self.sd = fit.mean(axis=0), fit.std(axis=0)
        self.sd[self.sd < 1e-6] = 1.0
        self.avg_sd = self.sd[0]

    def transform(self, df):
        return (df[FEATURES].to_numpy(float) - self.mean) / self.sd


def split_origins(n_rows, split):
    """Forecast origins o (first target row) whose targets o..o+HORIZON-1 fall in the split."""
    ranges = {'train': (INPUT_DAYS, TRAIN_END - HORIZON), 'val': (TRAIN_END, VAL_END - HORIZON),
              'test': (VAL_END, n_rows - HORIZON), 'all': (INPUT_DAYS, n_rows - HORIZON)}
    lo, hi = ranges[split]
    return list(range(max(lo, INPUT_DAYS), hi + 1))


def make_windows(x, avg, avg_sd, origins):
    """Inputs x[o-INPUT_DAYS:o]; target = future averages minus the anchor, in units of the archer's SD."""
    X = np.stack([x[o - INPUT_DAYS:o] for o in origins]).astype(np.float32)
    anchors = np.array([avg[o - ANCHOR_DAYS:o].mean() for o in origins])
    y = np.stack([(avg[o:o + HORIZON] - anchors[i]) / avg_sd for i, o in enumerate(origins)]).astype(np.float32)
    return X, y, anchors


# --- Model -----------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    def __init__(self, n_features=len(FEATURES), hidden=HIDDEN, layers=2, dropout=DROPOUT, horizon=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1]))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(X, y, X_val=None, y_val=None, epochs=None, max_epochs=300, patience=30, lr=1e-3, batch=32,
                hidden=HIDDEN, dropout=DROPOUT):
    """Train with early stopping on the validation set, or for a fixed number of epochs."""
    set_seed()
    model = LSTMForecaster(hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    X, y = torch.tensor(X), torch.tensor(y)
    has_val = X_val is not None and len(X_val)
    if has_val: X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
    history, best, best_state, best_epoch = [], math.inf, None, 0
    gen = torch.Generator().manual_seed(SEED)
    for epoch in range(1, (epochs or max_epochs) + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(X), generator=gen).split(batch):
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        row = dict(epoch=epoch, train_loss=total / len(X))
        if has_val:
            model.eval()
            with torch.no_grad(): row['val_loss'] = loss_fn(model(X_val), y_val).item()
            if row['val_loss'] < best:
                best, best_epoch = row['val_loss'], epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        history.append(row)
        if has_val and epochs is None and epoch - best_epoch >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, history, (best_epoch if has_val else len(history))


def mc_predict(model, X, samples=MC_SAMPLES):
    """Monte Carlo dropout: mean and SD of repeated stochastic predictions (model uncertainty)."""
    torch.manual_seed(SEED)
    model.train()  # keep dropout active
    with torch.no_grad():
        draws = torch.stack([model(torch.tensor(X)) for _ in range(samples)]).numpy()
    model.eval()
    return draws.mean(axis=0), draws.std(axis=0)


Z_80 = 1.2816  # two-sided 80 % interval of a normal distribution


def day_noise_sd(history):
    """Day-to-day spread of the session average, from consecutive differences so a
    slow trend does not inflate it (for independent noise, Var(diff) = 2 * sigma^2)."""
    return float(np.std(np.diff(history)) / math.sqrt(2))


def prediction_interval(anchor, mean, sd, scale, history):
    """80 % interval for a single future session: model uncertainty (MC dropout)
    combined with the archer's own day-to-day variation."""
    centre = anchor + mean * scale
    half = Z_80 * np.sqrt((sd * scale) ** 2 + day_noise_sd(history) ** 2)
    return centre, centre - half, centre + half


# --- Baselines and metrics ------------------------------------------------------------

def baseline_forecasts(avg, o):
    last7, last30 = avg[o - 7:o].mean(), avg[o - 30:o].mean()
    hist = avg[max(0, o - 90):o]
    slope, intercept = np.polyfit(np.arange(len(hist)), hist, 1)
    trend = intercept + slope * np.arange(len(hist), len(hist) + HORIZON)
    return {'Persistence (last 7)': np.full(HORIZON, last7), 'Moving average (last 30)': np.full(HORIZON, last30),
            'Linear trend (last 90)': trend}


def error_metrics(pred, actual):
    pred, actual = np.asarray(pred), np.asarray(actual)
    err = pred - actual
    return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                month_error=float(np.abs(pred.mean(axis=-1) - actual.mean(axis=-1)).mean()))


def summarise(errors):
    """errors: {model: [(pred, actual), ...]} -> {model: metrics} averaged over all windows."""
    return {m: {k: round(v, 4) for k, v in error_metrics(np.array([p for p, _ in pairs]),
                                                         np.array([a for _, a in pairs])).items()}
            for m, pairs in errors.items()}


# --- Evaluation pipeline ---------------------------------------------------------------

def prepare(archers):
    """archers: list of dict(name, username, df). Adds scaler, normalised x and raw avg."""
    for a in archers:
        a['scaler'] = Scaler(a['df'], fit_rows=TRAIN_END)
        a['x'], a['avg'] = a['scaler'].transform(a['df']), a['df']['avg'].to_numpy(float)
    return archers


def stack_split(archers, split):
    parts = [make_windows(a['x'], a['avg'], a['scaler'].avg_sd, split_origins(len(a['df']), split)) for a in archers]
    parts = [p for p in parts if len(p[0])]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def oracle_series(a, origins):
    """Simulator's expected value for every target day, if the archer is one of the virtual students."""
    import seed_synthetic_data as sim
    if a['username'] not in {p['username'] for p in sim.PROFILES}: return None
    offset = (a['df']['date'].iloc[0].date() - sim.START_DATE).days
    days = sorted({o + h for o in origins for h in range(HORIZON)})
    expected = {t: sim.expected_day_average(a['username'], t + offset) for t in days}
    return [np.array([expected[o + h] for h in range(HORIZON)]) for o in origins]


def test_errors(model, a, include_oracle=True, split='test'):
    """Forecast every origin of one archer's split with the LSTM and the baselines."""
    origins = split_origins(len(a['df']), split)
    X, _, anchors = make_windows(a['x'], a['avg'], a['scaler'].avg_sd, origins)
    mean, spread = mc_predict(model, X)
    errors, bands, inside = {'LSTM': []}, [], 0
    for i, o in enumerate(origins):
        actual = a['avg'][o:o + HORIZON]
        centre, low, high = prediction_interval(anchors[i], mean[i], spread[i], a['scaler'].avg_sd,
                                                a['avg'][o - INPUT_DAYS:o])
        errors['LSTM'].append((centre, actual))
        bands.append((low, high))
        inside += int(((actual >= low) & (actual <= high)).sum())
        for name, pred in baseline_forecasts(a['avg'], o).items(): errors.setdefault(name, []).append((pred, actual))
    oracle = oracle_series(a, origins) if include_oracle else None
    if oracle:
        errors['Oracle (simulator expectation)'] = [(oracle[i], a['avg'][o:o + HORIZON]) for i, o in enumerate(origins)]
    # The last test origin is kept as a worked example for the evaluation chart
    o, i = origins[-1], len(origins) - 1
    examples = dict(dates=[d.date().isoformat() for d in a['df']['date'].iloc[o - INPUT_DAYS:o + HORIZON]],
                    history=[round(v, 3) for v in a['avg'][o - INPUT_DAYS:o]],
                    actual=[round(v, 3) for v in a['avg'][o:o + HORIZON]],
                    forecasts={m: [round(float(v), 3) for v in pairs[i][0]] for m, pairs in errors.items()},
                    lower=[round(float(v), 3) for v in bands[i][0]], upper=[round(float(v), 3) for v in bands[i][1]])
    coverage = dict(inside=inside, total=len(origins) * HORIZON)
    return errors, examples, coverage


def evaluate(archers, log=print):
    """Train/validate/test the shared model, run leave-one-archer-out, and return
    the results plus the final model retrained on all data."""
    archers = prepare(archers)
    X_tr, y_tr = stack_split(archers, 'train')
    X_va, y_va = stack_split(archers, 'val')
    log(f'Training on {len(X_tr)} windows, validating on {len(X_va)}')
    model, history, best_epoch = train_model(X_tr, y_tr, X_va, y_va)
    val_pairs = [p for a in archers for p in test_errors(model, a, include_oracle=False, split='val')[0]['LSTM']]
    val_mae = summarise({'LSTM': val_pairs})['LSTM']['mae']
    log(f'Best epoch {best_epoch}, validation loss {min(h["val_loss"] for h in history):.4f}, validation MAE {val_mae:.4f}')

    per_archer, examples, overall, per_archer_cov, inside, total = {}, {}, {}, {}, 0, 0
    for a in archers:
        errors, examples[a['name']], cov = test_errors(model, a)
        per_archer[a['name']] = summarise(errors)
        per_archer_cov[a['name']] = round(cov['inside'] / cov['total'] * 100, 1)
        inside, total = inside + cov['inside'], total + cov['total']
        for m, pairs in errors.items(): overall.setdefault(m, []).extend(pairs)
    overall = summarise(overall)
    # Share of real sessions that fell inside the 80 % interval (should be close to 80)
    coverage = dict(overall=round(inside / total * 100, 1), per_archer=per_archer_cov)
    for m, v in overall.items(): log(f'  {m:<30} MAE {v["mae"]:.3f}  RMSE {v["rmse"]:.3f}  month error {v["month_error"]:.3f}')
    log(f'  80 % interval coverage on test: {coverage["overall"]} % ({coverage["per_archer"]})')

    # Leave-one-archer-out: train on the others, test on an archer the model has never seen
    loao = {}
    for held in archers:
        others = [a for a in archers if a is not held]
        Xo, yo = stack_split(others, 'train')
        Xv, yv = stack_split(others, 'val')
        m, _, _ = train_model(Xo, yo, Xv, yv)
        errors, _, _ = test_errors(m, held, include_oracle=False)
        loao[held['name']] = dict(unseen=summarise({'LSTM': errors['LSTM']})['LSTM'], seen=per_archer[held['name']]['LSTM'])
        log(f'  unseen {held["name"]:<8} MAE {loao[held["name"]]["unseen"]["mae"]:.3f} (seen {loao[held["name"]]["seen"]["mae"]:.3f})')

    # Final model: all data, same number of epochs, scalers fitted on the full history
    final_archers = []
    for a in archers:
        scaler = Scaler(a['df'])
        final_archers.append(dict(a, scaler=scaler, x=scaler.transform(a['df'])))
    X_all, y_all = stack_split(final_archers, 'all')
    final_model, _, _ = train_model(X_all, y_all, epochs=best_epoch)

    results = dict(config=dict(input_days=INPUT_DAYS, horizon=HORIZON, anchor_days=ANCHOR_DAYS, features=FEATURES,
                               train_end=TRAIN_END, val_end=VAL_END, mc_samples=MC_SAMPLES, best_epoch=best_epoch,
                               hidden=HIDDEN, dropout=DROPOUT,
                               val_mae=val_mae,
                               train_windows=len(X_tr), val_windows=len(X_va), archers=[a['name'] for a in archers]),
                   history=[{k: round(v, 5) for k, v in h.items()} for h in history],
                   overall=overall, per_archer=per_archer, coverage=coverage, leave_one_out=loao, examples=examples)
    return results, final_model


# --- Saving, loading and live forecasts ----------------------------------------------------

def save(model, results):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(dict(state=model.state_dict(), best_epoch=results['config']['best_epoch'],
                    hidden=HIDDEN, dropout=DROPOUT), MODEL_PATH)
    with open(EVAL_PATH, 'w', encoding='utf-8') as f: json.dump(results, f, indent=1)


def load_model():
    if not os.path.exists(MODEL_PATH): return None
    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    model = LSTMForecaster(hidden=checkpoint['hidden'], dropout=checkpoint['dropout'])
    model.load_state_dict(checkpoint['state'])
    model.eval()
    return model


def load_evaluation():
    if not os.path.exists(EVAL_PATH): return None
    with open(EVAL_PATH, encoding='utf-8') as f: return json.load(f)


def forecast(model, df):
    """Next HORIZON sessions after the last one in df (assumed to be one per day).
    Returns None when there are fewer than INPUT_DAYS sessions."""
    if model is None or len(df) < INPUT_DAYS: return None
    scaler = Scaler(df)
    x, avg = scaler.transform(df), df['avg'].to_numpy(float)
    anchor = avg[-ANCHOR_DAYS:].mean()
    mean, spread = mc_predict(model, x[-INPUT_DAYS:][None].astype(np.float32))
    centre, low, high = prediction_interval(anchor, mean[0], spread[0], scaler.avg_sd, avg[-INPUT_DAYS:])
    last = df['date'].iloc[-1]
    return [dict(date=(last + timedelta(days=h + 1)).date(), predicted=round(float(centre[h]), 3),
                 lower=round(float(low[h]), 3), upper=round(float(high[h]), 3)) for h in range(HORIZON)]
