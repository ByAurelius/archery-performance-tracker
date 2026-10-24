"""Turns the statistics from analytics.py into plain-language labels and sentences
for archers and coaches.

Every threshold used for a label is defined here, so the wording can be traced
back to a number (the thesis documents these choices in the technical section).
"""
import math

import numpy as np

import analytics

RING_WIDTH_CM = 6.1

# Day-to-day spread of the session average (points per arrow, trend removed)
CONSISTENCY_LEVELS = [(0.15, 'Very steady', 'good'), (0.30, 'Steady', 'good'),
                      (0.50, 'Variable', 'warn'), (math.inf, 'Very variable', 'warn')]
TREND_THRESHOLD = 0.05      # points per arrow per month
CENTRED_CM = 2.0            # drift below this counts as centred
FATIGUE_LEVELS = [(0.10, 'No real drop', 'good'), (0.25, 'Small drop', 'info'), (math.inf, 'Clear drop', 'warn')]

# Conditions compared in "What affects your score?": (column, name, low group, high group, unit)
FACTORS = [
    ('wind', 'Wind', ('calm days', lambda v: v <= 8), ('windy days', lambda v: v >= 15), 'km/h'),
    ('fatigue_level', 'Tiredness', ('fresh days', lambda v: v <= 3), ('tired days', lambda v: v >= 7), '/10'),
    ('focus_level', 'Focus', ('distracted days', lambda v: v <= 4), ('focused days', lambda v: v >= 7), '/10'),
    ('temp', 'Temperature', ('cold days', lambda v: v <= 5), ('warm days', lambda v: v >= 15), '°C'),
]


def per_session(points_per_arrow, arrows=100):
    return None if points_per_arrow is None else round(points_per_arrow * arrows)


def consistency(day_to_day):
    for limit, label, tone in CONSISTENCY_LEVELS:
        if day_to_day is not None and day_to_day < limit:
            return dict(label=label, tone=tone,
                        text=f'Your session average usually moves about ±{day_to_day:.2f} points per arrow from day to day.')
    return dict(label='–', tone='info', text='')


def trend(per_month):
    if per_month is None: return dict(label='Not enough data', tone='info', icon='–', text='')
    if per_month > TREND_THRESHOLD:
        return dict(label='Improving', tone='good', icon='📈',
                    text=f'Over the last 3 months you gained about {per_month:.2f} points per arrow each month.')
    if per_month < -TREND_THRESHOLD:
        return dict(label='Declining', tone='warn', icon='📉',
                    text=f'Over the last 3 months you lost about {abs(per_month):.2f} points per arrow each month.')
    return dict(label='Stable', tone='info', icon='➡️', text='Your level has stayed about the same over the last 3 months.')


def grouping(r50_mm):
    """Half the arrows land inside a circle of this radius; compare it with a target ring."""
    r_cm = r50_mm / 10
    ring = max(1, 11 - math.ceil(r_cm / RING_WIDTH_CM))
    return dict(cm=round(r_cm, 1), ring=ring,
                text=f'Half of your arrows land within {r_cm:.0f} cm of your group centre, about the size of the {ring} ring.')


def drift(mpi_x, mpi_y):
    """Which way the group centre sits from the middle of the target."""
    dist = math.hypot(mpi_x, mpi_y) / 10
    if dist < CENTRED_CM:
        return dict(label='Well centred', tone='good', arrow='🎯', cm=round(dist, 1),
                    text='Your arrows are centred on the middle of the target.')
    parts = []
    if abs(mpi_x) / 10 >= CENTRED_CM / 2: parts.append(f"{abs(mpi_x) / 10:.0f} cm {'right' if mpi_x > 0 else 'left'}")
    if abs(mpi_y) / 10 >= CENTRED_CM / 2: parts.append(f"{abs(mpi_y) / 10:.0f} cm {'high' if mpi_y > 0 else 'low'}")
    angle = math.degrees(math.atan2(mpi_y, mpi_x))
    arrow = ['➡️', '↗️', '⬆️', '↖️', '⬅️', '↙️', '⬇️', '↘️'][round(angle / 45) % 8]
    return dict(label=' and '.join(parts), tone='warn' if dist >= 4 else 'info', arrow=arrow, cm=round(dist, 1),
                text=f"Your arrows tend to land {' and '.join(parts)} of the centre."
                     + (' A sight adjustment may help.' if dist >= 4 else ''))


def fatigue(first_half, second_half):
    if first_half is None or second_half is None: return dict(label='–', tone='info', text='')
    drop = first_half - second_half
    for limit, label, tone in FATIGUE_LEVELS:
        if drop < limit:
            lost = per_session(drop, 50)
            return dict(label=label, tone=tone, drop=round(drop, 2), session_points=lost,
                        text=(f'In the second half of a session you score {drop:.2f} points per arrow less, '
                              f'about {lost} points over the last 50 arrows.') if drop >= 0.10 else
                             'Your score holds up well until the end of the session.')


EFFECT_P = 0.01   # stricter than 0.05 because several conditions are tested per archer
EFFECT_LEVELS = [(5, 'No clear effect'), (15, 'Small effect'), (40, 'Moderate effect'), (math.inf, 'Strong effect')]


def effect_strength(p, diff):
    """Plain label: the effect must be statistically clear (detrended p) and its size,
    in points per 100 arrows, decides how strong it is called."""
    if p is None or p >= EFFECT_P: return 'No clear effect'
    points = abs(diff) * 100
    return next(label for limit, label in EFFECT_LEVELS if points < limit)


def factor_effects(sessions, correlations):
    """For each condition: how much better or worse than usual you score in each group.
    Uses the day-to-day variation (trend removed), so seasons and learning do not mix in."""
    s = sessions.sort_values('date')
    residual = analytics.detrend(s['avg'])
    by_key = {f['key']: f for f in correlations['factors']}
    out = []
    for col, name, (low_name, low_test), (high_name, high_test), unit in FACTORS:
        values = s[col].astype(float)
        low, high = residual[low_test(values)], residual[high_test(values)]
        if len(low) < 5 or len(high) < 5: continue
        diff = high.mean() - low.mean()
        f = by_key.get(col, {})
        strength = effect_strength(f.get('p_detrended'), diff)
        if strength == 'No clear effect':
            text = f'No clear difference between {low_name} and {high_name}.'
        else:
            better, worse = (high_name, low_name) if diff > 0 else (low_name, high_name)
            text = (f'On {better} you score about {abs(diff):.2f} points per arrow more than on {worse} '
                    f'(≈ {per_session(abs(diff))} points per 100 arrows).')
        out.append(dict(key=col, name=name, strength=strength, text=text, low_name=low_name, high_name=high_name,
                        low=round(float(low.mean()), 3), high=round(float(high.mean()), 3),
                        n_low=int(len(low)), n_high=int(len(high)), diff=round(float(diff), 3),
                        tone='info' if strength == 'No clear effect' else 'warn'))
    return out


def highlights(report, sessions, forecast=None):
    """3-5 short sentences with the most useful findings, most important first."""
    s, out = report['summary'], []
    change = analytics.first_last_change(sessions)
    days = (sessions['date'].max() - sessions['date'].min()).days
    if change is not None and days >= 59:
        if change > TREND_THRESHOLD:
            out.append(('good', f'You improved by {change:.2f} points per arrow (≈ {per_session(change)} points per '
                                f'100 arrows) from your first month to your last month.'))
        elif change < -TREND_THRESHOLD:
            out.append(('warn', f'Your last month was {abs(change):.2f} points per arrow lower than your first month.'))
    d = drift(s['mpi_x'], s['mpi_y'])
    if d['tone'] == 'warn': out.append(('warn', d['text']))
    fat = report['fatigue']
    f = fatigue(fat['first_half'], fat['second_half'])
    if f['tone'] == 'warn':
        out.append(('warn', f['text'] + ' Endurance training could help.'))
    effects = [e for e in report.get('effects', []) if e['strength'] in ('Strong effect', 'Moderate effect')]
    if effects:
        e = max(effects, key=lambda e: abs(e['diff']))
        out.append(('info', f"{e['name']} matters most for you: {e['text'][0].lower()}{e['text'][1:]}"))
    if forecast:
        out.append(('info', f"Prediction for the next 30 days: about {forecast['month_mean']:.2f} points per arrow "
                            f"(≈ {per_session(forecast['month_mean'])} per 100 arrows)."))
    c = consistency(s['day_to_day'])
    if c['label'] == 'Very variable':
        out.append(('warn', 'Your results change a lot from day to day; a fixed pre-shot routine can make them steadier.'))
    return out[:5]


def plain_report(report, sessions):
    """Everything the plain-language page needs, built from the full analytics report."""
    s = report['summary']
    report['effects'] = factor_effects(sessions, report['correlations'])
    fat = report['fatigue']
    plain = dict(
        session_avg=per_session(s['avg_arrow']),
        trend=trend(s['recent_trend_per_30']),
        consistency=consistency(s['day_to_day']),
        grouping=grouping(s['r50']),
        drift=drift(s['mpi_x'], s['mpi_y']),
        fatigue=fatigue(fat['first_half'], fat['second_half']),
        effects=report['effects'],
        monthly=monthly_overview(sessions),
    )
    plain['highlights'] = highlights(report, sessions, report.get('forecast'))
    return plain


def monthly_overview(sessions):
    """Average, best day and worst day for each month (in points per arrow)."""
    rows = []
    for month, s in sessions.dropna(subset=['avg']).groupby(sessions['date'].dt.to_period('M')):
        rows.append(dict(month=month.strftime('%b %Y'), avg=round(float(s['avg'].mean()), 3),
                         best=round(float(s['avg'].max()), 3), worst=round(float(s['avg'].min()), 3), days=len(s)))
    return rows


def model_reliability(evaluation):
    """Plain-language summary of the forecast model evaluation."""
    if not evaluation: return None
    o = evaluation['overall']
    lstm, simple = o['LSTM'], o['Moving average (last 30)']
    oracle = next((v for k, v in o.items() if k.startswith('Oracle')), None)
    gain = (simple['mae'] - lstm['mae']) / simple['mae'] * 100
    return dict(
        session_error=per_session(lstm['mae']),
        month_error=round(lstm['month_error'], 2),
        vs_simple=('more accurate' if gain > 1 else 'about as accurate' if gain > -1 else 'less accurate'),
        gain_pct=round(abs(gain)),
        near_limit=bool(oracle and lstm['rmse'] <= oracle['rmse'] * 1.05),
        coverage=evaluation.get('coverage', {}).get('overall'),
        unseen_ok=all(abs(v['unseen']['mae'] - v['seen']['mae']) <= 0.02 for v in evaluation['leave_one_out'].values()),
        per_archer={name: per_session(m['LSTM']['mae']) for name, m in evaluation['per_archer'].items()},
    )
