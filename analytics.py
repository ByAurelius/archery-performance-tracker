"""Statistical analysis of an archer's training data.

The loaders read the database into pandas DataFrames; every other function is a
pure computation on those frames, so it can be unit-tested without a database.

Conventions: arrow coordinates are in mm from the target centre (x right, y up);
"session average" is the mean arrow score of one training day (0-10).
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

TARGET_RADIUS_MM = 610


# --- Loading -----------------------------------------------------------------

def parse_number(text):
    """'12.3 km/h' -> 12.3, '-' or None -> NaN."""
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return math.nan


def load_sessions(engine, archer_id):
    df = pd.read_sql(
        'SELECT id AS session_id, date, total_score, arrows_shot, fatigue_level, focus_level, '
        'wind_speed, weather_temp FROM score WHERE archer_id = :archer_id ORDER BY date, id',
        engine, params={'archer_id': archer_id})
    df['date'] = pd.to_datetime(df['date'])
    df['wind'] = df['wind_speed'].map(parse_number)
    df['temp'] = df['weather_temp'].map(parse_number)
    df['avg'] = df['total_score'] / df['arrows_shot'].where(df['arrows_shot'] > 0)
    return df.drop(columns=['wind_speed', 'weather_temp'])


def load_arrows(engine, archer_id):
    df = pd.read_sql(
        'SELECT a.x_mm, a.y_mm, a.score, a.end_no, a.arrow_no, s.arrows_per_end, s.id AS session_id, s.date '
        'FROM arrow a JOIN score s ON a.score_id = s.id WHERE s.archer_id = :archer_id',
        engine, params={'archer_id': archer_id})
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_plans(engine, archer_id):
    df = pd.read_sql('SELECT date_assigned, is_completed FROM training_plan WHERE athlete_id = :archer_id',
                     engine, params={'archer_id': archer_id})
    df['date_assigned'] = pd.to_datetime(df['date_assigned'], errors='coerce')
    df['is_completed'] = df['is_completed'].astype(bool)
    return df


# --- Helpers -------------------------------------------------------------------

def filter_period(sessions, arrows, days):
    """Keep the last `days` days of data (counted back from the latest session)."""
    if not days or sessions.empty: return sessions, arrows
    cutoff = sessions['date'].max() - pd.Timedelta(days=days - 1)
    return sessions[sessions['date'] >= cutoff], arrows[arrows['date'] >= cutoff]


def _clean(value, digits=2):
    """Round for display; NaN/inf -> None so it serialises to JSON null."""
    if value is None or not np.isfinite(value): return None
    return round(float(value), digits)


def group_stats(x, y):
    """Mean point of impact (MPI) and R50: median distance of the arrows from the MPI.

    The MPI measures accuracy (systematic offset from the centre), R50 measures
    precision (how tight the group is), both in mm.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0: return dict(mpi_x=None, mpi_y=None, mpi_dist=None, r50=None)
    mpi_x, mpi_y = x.mean(), y.mean()
    return dict(mpi_x=_clean(mpi_x, 1), mpi_y=_clean(mpi_y, 1), mpi_dist=_clean(math.hypot(mpi_x, mpi_y), 1),
                r50=_clean(np.median(np.hypot(x - mpi_x, y - mpi_y)), 1))


def linear_trend(sessions):
    """Least-squares line through the session averages against day number."""
    s = sessions.dropna(subset=['avg'])
    if len(s) < 3 or s['date'].nunique() < 2: return None
    days = (s['date'] - s['date'].min()).dt.days.to_numpy()
    fit = stats.linregress(days, s['avg'].to_numpy())
    return dict(slope_per_30=_clean(fit.slope * 30, 3), intercept=fit.intercept, slope=fit.slope,
                r2=_clean(fit.rvalue ** 2, 3), p=_clean(fit.pvalue, 4), start=s['date'].min())


def window_mean(sessions, start, end):
    """Mean session average for start <= date < end."""
    s = sessions[(sessions['date'] >= start) & (sessions['date'] < end)]['avg'].dropna()
    return s.mean() if len(s) else math.nan


# --- Analyses ------------------------------------------------------------------

def change_last_30(sessions):
    """Mean session average of the last 30 days minus that of the 30 days before."""
    last = sessions['date'].max()
    recent = window_mean(sessions, last - pd.Timedelta(days=29), last + pd.Timedelta(days=1))
    previous = window_mean(sessions, last - pd.Timedelta(days=59), last - pd.Timedelta(days=29))
    return _clean(recent - previous)


def summary(sessions, arrows):
    best = sessions.loc[sessions['total_score'].idxmax()]
    trend = linear_trend(sessions)
    return dict(
        sessions=len(sessions), arrows=len(arrows),
        avg_arrow=_clean(arrows['score'].mean()),
        change_30=change_last_30(sessions),
        pct_10=_clean((arrows['score'] == 10).mean() * 100, 1),
        pct_miss=_clean((arrows['score'] == 0).mean() * 100, 1),
        consistency=_clean(sessions['avg'].std(), 3),
        trend_per_30=trend['slope_per_30'] if trend else None,
        trend_r2=trend['r2'] if trend else None,
        trend_p=trend['p'] if trend else None,
        best_date=best['date'].date().isoformat(), best_total=int(best['total_score']),
        **group_stats(arrows['x_mm'], arrows['y_mm']),
    )


def progress(sessions):
    s = sessions.set_index('date')['avg']
    trend = linear_trend(sessions)
    trend_line = []
    if trend:
        days = (s.index - trend['start']).days
        trend_line = [_clean(trend['intercept'] + trend['slope'] * d, 3) for d in days]
    return dict(
        dates=[d.date().isoformat() for d in s.index],
        avg=[_clean(v, 3) for v in s],
        ma7=[_clean(v, 3) for v in s.rolling(7, min_periods=7).mean()],
        ma30=[_clean(v, 3) for v in s.rolling(30, min_periods=30).mean()],
        trend=trend_line,
    )


def score_distribution(arrows):
    """Share of each ring value (0 = miss ... 10). With 60+ days of data the first
    and last 30 days are returned separately to show the change."""
    def shares(a):
        counts = a['score'].value_counts().reindex(range(11), fill_value=0)
        return [_clean(v, 2) for v in counts / max(len(a), 1) * 100]

    first, last = arrows['date'].min(), arrows['date'].max()
    if pd.isna(first) or (last - first).days < 59:
        return dict(labels=['M'] + [str(i) for i in range(1, 11)], all=shares(arrows))
    return dict(labels=['M'] + [str(i) for i in range(1, 11)],
                first=shares(arrows[arrows['date'] < first + pd.Timedelta(days=30)]),
                last=shares(arrows[arrows['date'] > last - pd.Timedelta(days=30)]))


def impact_heatmap(arrows, bins=30):
    """2-D histogram of impacts over the target face, normalised to 0-1.
    Row 0 is the top of the target."""
    counts, _, _ = np.histogram2d(arrows['y_mm'], arrows['x_mm'], bins=bins,
                                  range=[[-TARGET_RADIUS_MM, TARGET_RADIUS_MM]] * 2)
    counts = counts[::-1]  # histogram rows run bottom-up; flip so row 0 is the top
    peak = counts.max()
    return dict(bins=bins, cells=(counts / peak if peak else counts).round(3).tolist(),
                **group_stats(arrows['x_mm'], arrows['y_mm']))


def monthly_boxplot(sessions):
    """Tukey box plot of the session averages per calendar month."""
    rows = []
    for month, s in sessions.dropna(subset=['avg']).groupby(sessions['date'].dt.to_period('M')):
        v = s['avg'].to_numpy()
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        inside = v[(v >= low) & (v <= high)]
        rows.append(dict(month=str(month), n=len(v), q1=_clean(q1, 3), median=_clean(med, 3), q3=_clean(q3, 3),
                         whisker_low=_clean(inside.min(), 3), whisker_high=_clean(inside.max(), 3),
                         outliers=[_clean(o, 3) for o in v[(v < low) | (v > high)]]))
    return rows


def in_session_fatigue(arrows):
    """Average score per end, and the first half of each session vs the second half."""
    if arrows.empty: return dict(ends=[], end_avg=[], first_half=None, second_half=None, drop=None)
    a = arrows.assign(seq=(arrows['end_no'] - 1) * arrows['arrows_per_end'] + arrows['arrow_no'])
    a['second'] = a['seq'] > a.groupby('session_id')['seq'].transform('max') / 2
    by_end = a.groupby('end_no')['score'].mean()
    first, second = a.loc[~a['second'], 'score'].mean(), a.loc[a['second'], 'score'].mean()
    return dict(ends=[int(e) for e in by_end.index], end_avg=[_clean(v, 3) for v in by_end],
                first_half=_clean(first, 3), second_half=_clean(second, 3), drop=_clean(first - second, 3))


CORRELATION_VARS = [('avg', 'Session average'), ('wind', 'Wind (km/h)'), ('temp', 'Temperature (°C)'),
                    ('fatigue_level', 'Fatigue'), ('focus_level', 'Focus')]


def _pearson(a, b):
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2: return math.nan, math.nan
    return stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])


def correlations(sessions):
    """Pearson r (with p-value) of each factor against the session average, plus
    the full correlation matrix.

    Because performance improves over the year while temperature and wind follow
    the seasons, a raw correlation can be driven by time alone. `r_detrended`
    correlates each factor with the session average minus its centred 31-day
    moving average, which removes the slow learning and seasonal trends.
    """
    s = sessions.sort_values('date')
    residual = s['avg'] - s['avg'].rolling(31, center=True, min_periods=1).mean()
    factors = []
    for col, label in CORRELATION_VARS[1:]:
        r, p = _pearson(s[col], s['avg'])
        r_d, p_d = _pearson(s[col] - s[col].rolling(31, center=True, min_periods=1).mean(), residual)
        pair = s[[col, 'avg']].dropna()
        factors.append(dict(key=col, label=label, r=_clean(r, 3), p=_clean(p, 4), n=len(pair),
                            r_detrended=_clean(r_d, 3), p_detrended=_clean(p_d, 4),
                            points=[[_clean(x, 2), _clean(y, 3)] for x, y in pair.to_numpy()]))
    cols = [c for c, _ in CORRELATION_VARS]
    matrix = sessions[cols].astype(float).corr()
    return dict(factors=factors, labels=[l for _, l in CORRELATION_VARS],
                matrix=[[_clean(v, 2) for v in row] for row in matrix.to_numpy()])


def accuracy_precision(arrows):
    """Per month: distance of the MPI from the centre (accuracy) and R50 (precision), in cm."""
    rows = []
    for month, a in arrows.groupby(arrows['date'].dt.to_period('M')):
        g = group_stats(a['x_mm'], a['y_mm'])
        rows.append(dict(month=str(month), mpi_cm=_clean(g['mpi_dist'] / 10, 2), r50_cm=_clean(g['r50'] / 10, 2)))
    return rows


def plan_effect(plans, sessions, window=14):
    """Completion rate, and the change in session average from the `window` days
    before a plan was assigned to the `window` days after, for completed vs. open plans."""
    if plans.empty: return None
    groups = {True: [], False: []}
    for _, plan in plans.dropna(subset=['date_assigned']).iterrows():
        d = plan['date_assigned']
        before = window_mean(sessions, d - pd.Timedelta(days=window), d)
        after = window_mean(sessions, d, d + pd.Timedelta(days=window))
        if np.isfinite(before) and np.isfinite(after): groups[plan['is_completed']].append(after - before)
    return dict(total=len(plans), completed=int(plans['is_completed'].sum()),
                rate=_clean(plans['is_completed'].mean() * 100, 1),
                change_completed=_clean(np.mean(groups[True]), 3) if groups[True] else None,
                change_open=_clean(np.mean(groups[False]), 3) if groups[False] else None,
                n_completed=len(groups[True]), n_open=len(groups[False]))


def full_report(sessions, arrows, plans):
    return dict(summary=summary(sessions, arrows), progress=progress(sessions),
                distribution=score_distribution(arrows), heatmap=impact_heatmap(arrows),
                boxplot=monthly_boxplot(sessions), fatigue=in_session_fatigue(arrows),
                correlations=correlations(sessions), accuracy_precision=accuracy_precision(arrows),
                plans=plan_effect(plans, sessions))


def first_last_change(sessions, days=30):
    """Mean session average of the last `days` days minus the first `days` days."""
    first, last = sessions['date'].min(), sessions['date'].max()
    return _clean(window_mean(sessions, last - pd.Timedelta(days=days - 1), last + pd.Timedelta(days=1))
                  - window_mean(sessions, first, first + pd.Timedelta(days=days)), 3)
