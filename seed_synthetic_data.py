"""Generates one year of reproducible training data for 5 virtual students.

Usage:  python seed_synthetic_data.py [--reset]

Simulation model (per student, per day, 100 arrows at 70 m on a 122 cm face):
  * Learning curve: the shot dispersion sigma shrinks exponentially,
        sigma_trend(t) = sigma_end + (sigma_start - sigma_end) * exp(-t / tau)
  * Daily form: sigma is multiplied by a log-normal factor exp(N(0, form_sd)).
  * Readiness: fatigue (1-10) widens and focus (1-10) narrows the group.
  * Weather: one shared club weather series (seasonal temperature and wind).
    The crosswind shifts the group sideways (partly corrected by the archer)
    and gusts add extra per-arrow spread.
  * In-session fatigue: sigma grows linearly from the first to the 100th arrow.
  * Each arrow is drawn as (x, y) ~ N(bias + wind shift, sigma) in mm from the
    centre, and scored with the same ring_score() the web app uses.
A fixed random seed makes every run produce exactly the same dataset.
"""
import argparse
import math
import random
from datetime import date, timedelta

from sqlalchemy import insert
from werkzeug.security import generate_password_hash

from app import app, db, Archer, Score, Arrow, ring_score

SEED = 42
START_DATE = date(2025, 10, 24)
DAYS = 365
ARROWS_PER_DAY = 100
ARROWS_PER_END = 6
DISTANCE_M = 70
STRING_LIFESPAN = 3000
STUDENT_PASSWORD = 'Student123!'

PROFILES = [
    # Fast-learning beginner
    dict(username='ali', first_name='Ali', age=17, sigma_start=220, sigma_end=110, tau=90, form_sd=0.08,
         fatigue_sens=0.03, focus_sens=0.02, drift=0.10, bias_x=0, bias_y=-15, wind_sens=4.0),
    # Steady intermediate
    dict(username='elif', first_name='Elif', age=21, sigma_start=130, sigma_end=95, tau=200, form_sd=0.06,
         fatigue_sens=0.02, focus_sens=0.02, drift=0.08, bias_x=-10, bias_y=0, wind_sens=3.0),
    # Plateaus after about four months
    dict(username='can', first_name='Can', age=23, sigma_start=125, sigma_end=100, tau=40, form_sd=0.06,
         fatigue_sens=0.02, focus_sens=0.02, drift=0.08, bias_x=5, bias_y=10, wind_sens=3.0),
    # Elite and very consistent
    dict(username='zeynep', first_name='Zeynep', age=25, sigma_start=75, sigma_end=60, tau=150, form_sd=0.04,
         fatigue_sens=0.01, focus_sens=0.01, drift=0.04, bias_x=0, bias_y=0, wind_sens=2.0),
    # Inconsistent, fatigue-sensitive, pulls to the right
    dict(username='mert', first_name='Mert', age=19, sigma_start=160, sigma_end=115, tau=180, form_sd=0.15,
         fatigue_sens=0.05, focus_sens=0.03, drift=0.20, bias_x=40, bias_y=0, wind_sens=4.5),
]

COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def clamp(value, low, high):
    return max(low, min(high, value))


def generate_weather(rng):
    """One weather record per day, shared by every student (same club)."""
    weather = []
    for t in range(DAYS):
        doy = (START_DATE + timedelta(days=t)).timetuple().tm_yday
        temp = 10 + 10 * math.sin(2 * math.pi * (doy - 110) / 365) + rng.gauss(0, 3)
        speed = max(0.0, 10 + 4 * math.cos(2 * math.pi * (doy - 15) / 365) + rng.gauss(0, 4))
        direction = rng.gauss(270, 60) % 360  # prevailing westerly
        weather.append(dict(temp=round(temp, 1), speed=round(speed, 1), direction=round(direction)))
    return weather


def simulate_day(rng, p, t, w, prev_fatigue):
    fatigue = clamp(round(rng.gauss(4.5, 1.7) + 0.3 * (prev_fatigue - 4.5)), 1, 10)
    focus = clamp(round(rng.gauss(6.5, 1.5) - 0.3 * (fatigue - 5)), 1, 10)

    sigma_trend = p['sigma_end'] + (p['sigma_start'] - p['sigma_end']) * math.exp(-t / p['tau'])
    sigma_day = (sigma_trend * math.exp(rng.gauss(0, p['form_sd']))
                 * (1 + p['fatigue_sens'] * (fatigue - 5))
                 * (1 - p['focus_sens'] * (focus - 5))
                 * (1 + 0.008 * w['speed']))

    # Shooting line faces north: a westerly wind (from 270 deg) pushes arrows to the right (+x)
    crosswind = -w['speed'] * math.sin(math.radians(w['direction']))
    shift_x = p['bias_x'] + 0.5 * p['wind_sens'] * crosswind
    gust_sd = 0.3 * p['wind_sens'] * w['speed']

    shots = []
    for k in range(ARROWS_PER_DAY):
        sigma = sigma_day * (1 + p['drift'] * k / (ARROWS_PER_DAY - 1))
        x = shift_x + rng.gauss(0, math.hypot(sigma, gust_sd))
        y = p['bias_y'] + rng.gauss(0, sigma)
        shots.append((round(x, 1), round(y, 1), ring_score(x, y)))
    return fatigue, focus, shots


def reset_synthetic():
    ids = [a.id for a in Archer.query.filter_by(is_synthetic=True)]
    if not ids: return
    score_ids = db.session.query(Score.id).filter(Score.archer_id.in_(ids))
    Arrow.query.filter(Arrow.score_id.in_(score_ids)).delete(synchronize_session=False)
    Score.query.filter(Score.archer_id.in_(ids)).delete(synchronize_session=False)
    Archer.query.filter(Archer.id.in_(ids)).delete(synchronize_session=False)
    db.session.commit()
    print(f'Removed {len(ids)} existing virtual students.')


def seed():
    weather = generate_weather(random.Random(SEED))

    for p in PROFILES:
        if Archer.query.filter_by(username=p['username']).first():
            print(f"{p['username']} already exists, skipping (use --reset to regenerate).")
            continue
        # A separate stream per student keeps each one's data identical regardless of the others
        rng = random.Random(f"{SEED}-{p['username']}")
        archer = Archer(username=p['username'], email=f"{p['username']}@demo.com",
                        password=generate_password_hash(STUDENT_PASSWORD), first_name=p['first_name'],
                        last_name='(virtual)', age=p['age'], is_synthetic=True,
                        current_string_arrows=(DAYS * ARROWS_PER_DAY) % STRING_LIFESPAN)
        db.session.add(archer)
        db.session.flush()

        sessions, prev_fatigue = [], 4.5
        for t in range(DAYS):
            w = weather[t]
            fatigue, focus, shots = simulate_day(rng, p, t, w, prev_fatigue)
            prev_fatigue = fatigue
            score = Score(date=START_DATE + timedelta(days=t), distance=DISTANCE_M,
                          arrows_shot=len(shots), total_score=sum(s[2] for s in shots),
                          arrows_per_end=ARROWS_PER_END, weather_temp=f"{w['temp']} °C",
                          wind_speed=f"{w['speed']} km/h",
                          wind_direction=f"{COMPASS[round(w['direction'] / 45) % 8]} ({w['direction']}°)",
                          fatigue_level=fatigue, focus_level=focus, archer_id=archer.id)
            sessions.append((score, shots))
        db.session.add_all(s for s, _ in sessions)
        db.session.flush()

        arrow_rows = []
        for score, shots in sessions:
            # Like the scorecard UI, arrows inside each end are listed highest score first
            for start in range(0, len(shots), ARROWS_PER_END):
                end = sorted(shots[start:start + ARROWS_PER_END], key=lambda s: -s[2])
                for j, (x, y, s) in enumerate(end):
                    arrow_rows.append(dict(end_no=start // ARROWS_PER_END + 1, arrow_no=j + 1,
                                           x_mm=x, y_mm=y, score=s, score_id=score.id))
        db.session.execute(insert(Arrow), arrow_rows)
        db.session.commit()

        first = sum(s.total_score for s, _ in sessions[:30]) / (30 * ARROWS_PER_DAY)
        last = sum(s.total_score for s, _ in sessions[-30:]) / (30 * ARROWS_PER_DAY)
        print(f"{p['first_name']:<7} {len(sessions)} sessions, {len(arrow_rows)} arrows, "
              f"avg arrow score {first:.2f} (first month) -> {last:.2f} (last month)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--reset', action='store_true', help='delete existing virtual students first')
    args = parser.parse_args()
    with app.app_context():
        if args.reset: reset_synthetic()
        seed()
