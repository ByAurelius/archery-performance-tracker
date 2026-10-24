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
  * Equipment: after a limb change (higher draw weight) the group opens up by
    up to 12 % and recovers over roughly two weeks.
  * Each arrow is drawn as (x, y) ~ N(bias + wind shift, sigma) in mm from the
    centre, and scored with the same ring_score() the web app uses.

Besides the arrows, each student also gets consistent supporting records:
  * Bow tuning logs that follow a per-student story (limb upgrades, tuning trials).
  * Sight marks for 18-70 m, re-marked after every limb change; a faster arrow
    (higher draw weight) needs less sight drop at long distance.
  * Coach training plans every two weeks, picked from a catalogue according to
    the student's needs, with a per-student completion rate.
A fixed random seed makes every run produce exactly the same dataset.
"""
import argparse
import math
import random
from datetime import date, timedelta

from sqlalchemy import insert
from werkzeug.security import generate_password_hash

from app import app, db, Archer, Score, Arrow, SightMark, TrainingPlan, BowTuning, ring_score

SEED = 42
START_DATE = date(2025, 10, 24)
DAYS = 365
ARROWS_PER_DAY = 100
ARROWS_PER_END = 6
DISTANCE_M = 70
STRING_LIFESPAN = 3000
STUDENT_PASSWORD = 'Student123!'
SIGHT_DISTANCES = [18, 30, 50, 60, 70]
PLAN_INTERVAL_DAYS = 14
LIMB_CHANGE_EFFECT = 0.12   # extra group spread right after a limb change
LIMB_CHANGE_RECOVERY = 5    # days (exponential decay constant)

# Each tuning event is (day, changed settings, note). Day 0 is the initial setup;
# a change of draw weight means new limbs.
PROFILES = [
    # Fast-learning beginner
    dict(username='ali', first_name='Ali', age=17, height=176, weight=64, draw_length=27.0,
         sigma_start=220, sigma_end=110, tau=90, form_sd=0.08,
         fatigue_sens=0.03, focus_sens=0.02, drift=0.10, bias_x=0, bias_y=-15, wind_sens=4.0,
         bow='WNS Elan 66"', sight_offset=0.15, windage=0.2, plan_rate=0.85,
         plan_focus=dict(technique=0.5, conditioning=0.3, spt=0.15, mental=0.05),
         tuning=[
             (0, dict(brace_height=21.5, tiller_upper=17.5, tiller_lower=17.0, nocking_point=5, draw_weight=26),
              'Initial club setup with 26 lb training limbs.'),
             (45, dict(nocking_point=6), 'Bare shafts landing low; raised the nocking point 1 mm.'),
             (140, dict(draw_weight=30), 'New 30 lb limbs installed. Groups opened up for the first week.'),
             (175, dict(brace_height=22.0), 'Brace height raised to reduce string slap on the arm guard.'),
             (270, dict(draw_weight=34), 'Moved up to 34 lb limbs before the outdoor season.'),
             (300, dict(tiller_upper=17.6), 'Small tiller tweak, bow now settles more quietly at full draw.'),
         ]),
    # Steady intermediate
    dict(username='elif', first_name='Elif', age=21, height=168, weight=58, draw_length=26.5,
         sigma_start=130, sigma_end=95, tau=200, form_sd=0.06,
         fatigue_sens=0.02, focus_sens=0.02, drift=0.08, bias_x=-10, bias_y=0, wind_sens=3.0,
         bow='Kinetic Sovren 68"', sight_offset=-0.1, windage=-0.1, plan_rate=0.9,
         plan_focus=dict(technique=0.25, conditioning=0.25, spt=0.25, mental=0.25),
         tuning=[
             (0, dict(brace_height=22.0, tiller_upper=18.0, tiller_lower=17.6, nocking_point=6, draw_weight=32),
              'Season setup.'),
             (90, dict(brace_height=22.3), 'Twisted the string 3 turns to reduce noise.'),
             (180, dict(draw_weight=34), 'New 34 lb limbs; paper test clean after one nocking adjustment.'),
             (250, dict(nocking_point=5), 'Nocking point lowered 1 mm, arrows flying cleaner.'),
         ]),
    # Plateaus after about four months
    dict(username='can', first_name='Can', age=23, height=182, weight=78, draw_length=28.5,
         sigma_start=125, sigma_end=100, tau=40, form_sd=0.06,
         fatigue_sens=0.02, focus_sens=0.02, drift=0.08, bias_x=5, bias_y=10, wind_sens=3.0,
         bow='Win&Win Wiawis ATF-DX 68"', sight_offset=0.05, windage=0.1, plan_rate=0.75,
         plan_focus=dict(technique=0.35, mental=0.35, spt=0.15, conditioning=0.15),
         tuning=[
             (0, dict(brace_height=22.5, tiller_upper=18.2, tiller_lower=17.8, nocking_point=6, draw_weight=36),
              'Season setup.'),
             (130, dict(brace_height=22.0), 'Scores flat for weeks; trying a lower brace height.'),
             (160, dict(nocking_point=7), 'Testing a higher nocking point.'),
             (195, dict(brace_height=22.8), 'Lower brace felt unstable, now testing 22.8 cm.'),
             (230, dict(tiller_upper=18.0), 'Reduced tiller difference after walk-back test.'),
             (300, dict(brace_height=22.5, nocking_point=6),
              'No measurable change from the trials; back to the original setup.'),
         ]),
    # Elite and very consistent
    dict(username='zeynep', first_name='Zeynep', age=25, height=171, weight=61, draw_length=27.5,
         sigma_start=75, sigma_end=60, tau=150, form_sd=0.04,
         fatigue_sens=0.01, focus_sens=0.01, drift=0.04, bias_x=0, bias_y=0, wind_sens=2.0,
         bow='Hoyt Formula Xi 70"', sight_offset=0.0, windage=0.0, plan_rate=0.95,
         plan_focus=dict(mental=0.4, spt=0.3, technique=0.15, conditioning=0.15),
         tuning=[
             (0, dict(brace_height=23.0, tiller_upper=18.4, tiller_lower=18.0, nocking_point=5, draw_weight=42),
              'Competition setup.'),
             (120, dict(brace_height=23.2), 'Fine tuning before the indoor finals.'),
             (260, dict(nocking_point=5.5), 'Half a millimetre up after bare shaft test at 50 m.'),
         ]),
    # Inconsistent, fatigue-sensitive, pulls to the right
    dict(username='mert', first_name='Mert', age=19, height=179, weight=72, draw_length=28.0,
         sigma_start=160, sigma_end=115, tau=180, form_sd=0.15,
         fatigue_sens=0.05, focus_sens=0.03, drift=0.20, bias_x=40, bias_y=0, wind_sens=4.5,
         bow='Uukha X-Factor 68"', sight_offset=-0.05, windage=0.0, plan_rate=0.6,
         plan_focus=dict(conditioning=0.35, spt=0.3, mental=0.2, technique=0.15),
         windage_fixes=[(60, -0.3), (150, -0.5), (240, -0.6)],
         tuning=[
             (0, dict(brace_height=22.0, tiller_upper=17.8, tiller_lower=17.6, nocking_point=6, draw_weight=34),
              'Season setup.'),
             (60, dict(tiller_upper=17.6), 'Groups pulling right; adjusted tiller and moved sight left.'),
             (100, dict(nocking_point=7), 'Still pulling right, testing a higher nocking point.'),
             (150, dict(), 'Fitted a stiffer plunger spring. Right pull is a release issue, not the bow.'),
             (220, dict(draw_weight=36), 'New 36 lb limbs.'),
             (300, dict(brace_height=22.4), 'Brace height raised; bow quieter.'),
         ]),
]

PLAN_CATALOGUE = {
    'conditioning': [
        ('Core stability circuit',
         '3 rounds: 45 s plank, 30 s side plank each side, 15 dead bugs. Rest 60 s between rounds.'),
        ('Shoulder resistance band routine',
         'External rotations, face pulls and band pull-aparts, 3 x 15 each, both arms.'),
        ('Aerobic base run', '30 min easy run at conversational pace, 3 times this week.'),
        ('Interval cardio', '6 x (2 min brisk run + 1 min walk) to improve heart-rate control.'),
    ],
    'spt': [
        ('SPT holds', '5 sets of 6 x 10 s full-draw holds with your own bow, 20 s rest between holds.'),
        ('Clicker endurance', '3 x 12 slow draws to the clicker without shooting. Focus on back tension.'),
        ('Heavy band draws', '4 x 15 draws with a heavy band, holding 3 s at anchor each time.'),
    ],
    'technique': [
        ('Blank bale: release focus',
         '150 arrows at 3 m, eyes closed every third end. Feel a relaxed, continuous release.'),
        ('Stance and alignment check',
         'Film yourself from behind for 3 ends and compare bow arm and draw elbow alignment.'),
        ('Short distance grouping', '6 ends at 30 m. Goal: every arrow inside the red.'),
        ('Follow-through drill', 'Hold your follow-through for 2 s after every shot, 100 arrows.'),
    ],
    'mental': [
        ('Visualisation session', '10 min every day: visualise a full shot sequence from nocking to follow-through.'),
        ('Breathing routine', 'Use 4-4-6 breathing before each end in this week\'s sessions.'),
        ('Competition simulation', 'Shoot a full 72-arrow 70 m round with scoring and a 2 min limit per end.'),
        ('Shot routine journal', 'After each session write 3 lines: what felt good, what to keep, what to change.'),
    ],
}

COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def clamp(value, low, high):
    return max(low, min(high, value))


def day_str(t):
    return (START_DATE + timedelta(days=t)).isoformat()


def limb_change_days(p):
    days, weight = [], None
    for day, changes, _ in p['tuning']:
        if 'draw_weight' in changes:
            if weight is not None and changes['draw_weight'] != weight: days.append(day)
            weight = changes['draw_weight']
    return days


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


def simulate_day(rng, p, t, w, prev_fatigue, limb_days):
    fatigue = clamp(round(rng.gauss(4.5, 1.7) + 0.3 * (prev_fatigue - 4.5)), 1, 10)
    focus = clamp(round(rng.gauss(6.5, 1.5) - 0.3 * (fatigue - 5)), 1, 10)

    sigma_trend = p['sigma_end'] + (p['sigma_start'] - p['sigma_end']) * math.exp(-t / p['tau'])
    equipment = 1 + sum(LIMB_CHANGE_EFFECT * math.exp(-(t - d) / LIMB_CHANGE_RECOVERY) for d in limb_days if t >= d)
    sigma_day = (sigma_trend * equipment * math.exp(rng.gauss(0, p['form_sd']))
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


def sight_elevation(distance, draw_weight, offset):
    # Sight tape reading: longer distance -> lower reading; a faster arrow drops less
    drop = 0.0686 * distance * (36 / draw_weight) ** 0.8
    return 8.2 - drop + offset


def format_sight(elevation, windage):
    side = 'R' if windage > 0 else 'L' if windage < 0 else ''
    return f'{elevation:.2f} / windage {abs(windage):.1f}{side}'


def build_tuning_logs(p, archer_id):
    logs, setup = [], {}
    for day, changes, note in p['tuning']:
        setup.update(changes)
        logs.append(BowTuning(date_recorded=day_str(day), bow_name=p['bow'],
                              brace_height=f"{setup['brace_height']} cm",
                              tiller_upper=f"{setup['tiller_upper']} cm", tiller_lower=f"{setup['tiller_lower']} cm",
                              nocking_point=f"+{setup['nocking_point']} mm",
                              draw_weight=f"{setup['draw_weight']} lbs", notes=note, archer_id=archer_id))
    return logs


def build_sight_marks(rng, p, archer_id):
    marks = []
    weights = {day: c['draw_weight'] for day, c, _ in p['tuning'] if 'draw_weight' in c}
    windage = p['windage']
    fixes = dict(p.get('windage_fixes', []))
    # Full re-mark at the start and a few days after each limb change, plus occasional 70 m fine tuning
    events = sorted([(0, 'full')] + [(d + 3, 'full') for d in limb_change_days(p)]
                    + [(d, 'windage') for d in fixes] + [(rng.randint(20, 120), 'fine'), (rng.randint(190, 330), 'fine')])
    draw_weight, fine = weights[0], 0.0
    for day, kind in events:
        draw_weight = weights[max(d for d in weights if d <= day)]
        if kind == 'windage': windage = fixes[day]
        if kind == 'fine': fine += rng.choice([-0.05, 0.05])
        distances = SIGHT_DISTANCES if kind == 'full' else [70]
        for distance in distances:
            elevation = sight_elevation(distance, draw_weight, p['sight_offset'] + fine) + rng.gauss(0, 0.02)
            marks.append(SightMark(date_recorded=day_str(day), distance=distance,
                                   setting=format_sight(elevation, windage), archer_id=archer_id))
    return marks


def build_training_plans(rng, p, archer_id):
    plans, last_title = [], None
    categories, weights = zip(*p['plan_focus'].items())
    for day in range(3, DAYS, PLAN_INTERVAL_DAYS):
        options = [o for o in PLAN_CATALOGUE[rng.choices(categories, weights)[0]] if o[0] != last_title]
        title, description = rng.choice(options)
        last_title = title
        # Plans from the last three weeks are still open
        completed = day < DAYS - 21 and rng.random() < p['plan_rate']
        plans.append(TrainingPlan(title=title, description=description, date_assigned=day_str(day),
                                  is_completed=completed, athlete_id=archer_id))
    return plans


def reset_synthetic():
    ids = [a.id for a in Archer.query.filter_by(is_synthetic=True)]
    if not ids: return
    score_ids = db.session.query(Score.id).filter(Score.archer_id.in_(ids))
    Arrow.query.filter(Arrow.score_id.in_(score_ids)).delete(synchronize_session=False)
    Score.query.filter(Score.archer_id.in_(ids)).delete(synchronize_session=False)
    SightMark.query.filter(SightMark.archer_id.in_(ids)).delete(synchronize_session=False)
    BowTuning.query.filter(BowTuning.archer_id.in_(ids)).delete(synchronize_session=False)
    TrainingPlan.query.filter(TrainingPlan.athlete_id.in_(ids)).delete(synchronize_session=False)
    Archer.query.filter(Archer.id.in_(ids)).delete(synchronize_session=False)
    db.session.commit()
    print(f'Removed {len(ids)} existing virtual students.')


def seed():
    weather = generate_weather(random.Random(SEED))

    for p in PROFILES:
        if Archer.query.filter_by(username=p['username']).first():
            print(f"{p['username']} already exists, skipping (use --reset to regenerate).")
            continue
        # Separate streams per student (and for arrows vs. records) keep each dataset independent
        rng = random.Random(f"{SEED}-{p['username']}")
        records_rng = random.Random(f"{SEED}-{p['username']}-records")
        archer = Archer(username=p['username'], email=f"{p['username']}@demo.com",
                        password=generate_password_hash(STUDENT_PASSWORD), first_name=p['first_name'],
                        last_name='(virtual)', age=p['age'], height=p['height'], weight=p['weight'],
                        draw_length=p['draw_length'], is_synthetic=True,
                        current_string_arrows=(DAYS * ARROWS_PER_DAY) % STRING_LIFESPAN)
        db.session.add(archer)
        db.session.flush()

        sessions, prev_fatigue, limb_days = [], 4.5, limb_change_days(p)
        for t in range(DAYS):
            w = weather[t]
            fatigue, focus, shots = simulate_day(rng, p, t, w, prev_fatigue, limb_days)
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

        tuning_logs = build_tuning_logs(p, archer.id)
        sight_marks = build_sight_marks(records_rng, p, archer.id)
        plans = build_training_plans(records_rng, p, archer.id)
        db.session.add_all(tuning_logs + sight_marks + plans)
        db.session.commit()

        first = sum(s.total_score for s, _ in sessions[:30]) / (30 * ARROWS_PER_DAY)
        last = sum(s.total_score for s, _ in sessions[-30:]) / (30 * ARROWS_PER_DAY)
        done = sum(pl.is_completed for pl in plans)
        print(f"{p['first_name']:<7} {len(sessions)} sessions, {len(arrow_rows)} arrows, "
              f"avg arrow score {first:.2f} -> {last:.2f} | {len(tuning_logs)} tuning logs, "
              f"{len(sight_marks)} sight marks, {len(plans)} plans ({done} done)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--reset', action='store_true', help='delete existing virtual students first')
    args = parser.parse_args()
    with app.app_context():
        if args.reset: reset_synthetic()
        seed()
