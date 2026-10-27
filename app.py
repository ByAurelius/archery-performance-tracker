import os
import json
import csv
import math
from io import StringIO
import secrets
import threading
from datetime import date, datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, make_response, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import analytics
import insights
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

load_dotenv()  # reads settings such as SECRET_KEY from the .env file
app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY')
if not app.secret_key:
    # Without a fixed key everyone is logged out whenever the app restarts
    app.secret_key = secrets.token_hex(32)
    print('WARNING: SECRET_KEY is not set. Copy .env.example to .env and add a key (see README).')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///archery.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# 122 cm WA target face: radius 610 mm, 10 rings of 61 mm each.
# The canvas draws it with a 200 px radius, so 1 px = 3.05 mm.
TARGET_RADIUS_MM = 610
RING_WIDTH_MM = 61
CANVAS_CENTER_PX = 200
MM_PER_PX = TARGET_RADIUS_MM / CANVAS_CENTER_PX


# ... (Database modelleri ayni kalacak) ...
class Archer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    is_coach = db.Column(db.Boolean, default=False)
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    weight = db.Column(db.Float, nullable=True)
    height = db.Column(db.Float, nullable=True)
    draw_length = db.Column(db.Float, nullable=True)
    current_string_arrows = db.Column(db.Integer, default=0)
    is_synthetic = db.Column(db.Boolean, default=False)
    scores = db.relationship('Score', backref='shooter', lazy=True)
    sight_marks = db.relationship('SightMark', backref='shooter', lazy=True)
    training_plans = db.relationship('TrainingPlan', backref='receiver', lazy=True)
    tuning_logs = db.relationship('BowTuning', backref='shooter', lazy=True)


class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, index=True)
    distance = db.Column(db.Integer, nullable=False)
    arrows_shot = db.Column(db.Integer, nullable=False)
    total_score = db.Column(db.Integer, nullable=False)
    image_file = db.Column(db.String(255), nullable=True)
    arrows_per_end = db.Column(db.Integer, default=6)
    weather_temp = db.Column(db.String(20), nullable=True)
    wind_speed = db.Column(db.String(20), nullable=True)
    wind_direction = db.Column(db.String(20), nullable=True)
    fatigue_level = db.Column(db.Integer, nullable=True)
    focus_level = db.Column(db.Integer, nullable=True)
    archer_id = db.Column(db.Integer, db.ForeignKey('archer.id'), nullable=False, index=True)
    arrows = db.relationship('Arrow', backref='training', lazy=True, cascade='all, delete-orphan',
                             order_by='Arrow.id')


class Arrow(db.Model):
    # One row per arrow. Coordinates are in mm from the target centre (x right, y up).
    id = db.Column(db.Integer, primary_key=True)
    end_no = db.Column(db.Integer, nullable=False)
    arrow_no = db.Column(db.Integer, nullable=False)
    x_mm = db.Column(db.Float, nullable=False)
    y_mm = db.Column(db.Float, nullable=False)
    score = db.Column(db.Integer, nullable=False)
    score_id = db.Column(db.Integer, db.ForeignKey('score.id'), nullable=False, index=True)


def ring_score(x_mm, y_mm):
    radius = (x_mm ** 2 + y_mm ** 2) ** 0.5
    if radius > TARGET_RADIUS_MM: return 0
    return max(1, min(10, math.ceil((TARGET_RADIUS_MM - radius) / RING_WIDTH_MM)))


class Forecast(db.Model):
    # LSTM forecast of the session average for one future day, with an 80 % band
    id = db.Column(db.Integer, primary_key=True)
    archer_id = db.Column(db.Integer, db.ForeignKey('archer.id'), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False)
    predicted = db.Column(db.Float, nullable=False)
    lower = db.Column(db.Float, nullable=False)
    upper = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)


class SightMark(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_recorded = db.Column(db.String(20), nullable=False)
    distance = db.Column(db.Integer, nullable=False)
    setting = db.Column(db.String(50), nullable=False)
    image_file = db.Column(db.String(255), nullable=True)
    archer_id = db.Column(db.ForeignKey('archer.id'), nullable=False)


class TrainingPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    date_assigned = db.Column(db.String(20), nullable=False)
    is_completed = db.Column(db.Boolean, default=False)
    athlete_id = db.Column(db.Integer, db.ForeignKey('archer.id'), nullable=False)


class BowTuning(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_recorded = db.Column(db.String(20), nullable=False)
    bow_name = db.Column(db.String(100), nullable=False)
    brace_height = db.Column(db.String(50), nullable=True)
    tiller_upper = db.Column(db.String(50), nullable=True)
    tiller_lower = db.Column(db.String(50), nullable=True)
    nocking_point = db.Column(db.String(50), nullable=True)
    draw_weight = db.Column(db.String(50), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    archer_id = db.Column(db.Integer, db.ForeignKey('archer.id'), nullable=False)


# --- ERROR HANDLERS ---
@app.errorhandler(404)
def page_not_found(e): return render_template('error.html', code=404, message="Page Not Found"), 404


@app.errorhandler(500)
def server_error(e): return render_template('error.html', code=500, message="Internal Server Error"), 500


# --- ROUTES ---
@app.route('/')
def home_page(): return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        hashed_password = generate_password_hash(request.form.get('password'))
        new_archer = Archer(username=request.form.get('username'), email=request.form.get('email'),
                            password=hashed_password, is_coach=(request.form.get('is_coach') == 'on'))
        db.session.add(new_archer)
        db.session.commit()
        return redirect(url_for('login_page'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        archer = Archer.query.filter_by(email=request.form.get('email')).first()
        if archer and check_password_hash(archer.password, request.form.get('password')):
            session['user_id'] = archer.id
            session['username'] = archer.username
            session['is_coach'] = archer.is_coach
            return redirect(url_for('coach_dashboard_page') if archer.is_coach else url_for('dashboard_page'))
        return "<h1>Invalid Email or Password!</h1>"
    return render_template('login.html')


@app.route('/profile', methods=['GET', 'POST'])
def profile_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    archer = Archer.query.get(session['user_id'])
    if request.method == 'POST':
        archer.first_name, archer.last_name = request.form.get('first_name'), request.form.get('last_name')
        archer.age, archer.weight = request.form.get('age'), request.form.get('weight')
        archer.height, archer.draw_length = request.form.get('height'), request.form.get('draw_length')
        db.session.commit()
        return redirect(url_for('profile_page'))
    return render_template('profile.html', archer=archer)


@app.route('/coach_dashboard')
def coach_dashboard_page():
    if 'user_id' in session and session.get('is_coach'):
        return render_template('coach_dashboard.html', username=session['username'],
                               athletes=Archer.query.filter_by(is_coach=False).all())
    return redirect(url_for('home_page'))


@app.route('/athlete/<int:athlete_id>')
def athlete_stats_page(athlete_id):
    if 'user_id' not in session or not session.get('is_coach'): return redirect(url_for('home_page'))
    athlete = Archer.query.get_or_404(athlete_id)
    if athlete.is_coach: return redirect(url_for('coach_dashboard_page'))
    return render_template('athlete_stats.html', athlete=athlete,
                           scores=Score.query.filter_by(archer_id=athlete.id).all(),
                           sight_marks=SightMark.query.filter_by(archer_id=athlete.id).order_by(
                               SightMark.date_recorded.desc(), SightMark.id.desc()).all(),
                           plans=TrainingPlan.query.filter_by(athlete_id=athlete.id).order_by(
                               TrainingPlan.date_assigned.desc(), TrainingPlan.id.desc()).all(),
                           tuning_logs=BowTuning.query.filter_by(archer_id=athlete.id).order_by(
                               BowTuning.date_recorded.desc(), BowTuning.id.desc()).all())


@app.route('/assign_plan/<int:athlete_id>', methods=['POST'])
def assign_plan(athlete_id):
    if 'user_id' not in session or not session.get('is_coach'): return redirect(url_for('home_page'))
    db.session.add(TrainingPlan(title=request.form.get('title'), description=request.form.get('description'),
                                date_assigned=request.form.get('date_assigned'), athlete_id=athlete_id))
    db.session.commit()
    return redirect(url_for('athlete_stats_page', athlete_id=athlete_id))


@app.route('/complete_plan/<int:plan_id>')
def complete_plan(plan_id):
    if 'user_id' not in session: return redirect(url_for('login_page'))
    plan = TrainingPlan.query.get_or_404(plan_id)
    if plan.athlete_id == session['user_id']:
        plan.is_completed = True
        db.session.commit()
    return redirect(url_for('dashboard_page'))


@app.route('/add_score', methods=['GET', 'POST'])
def add_score_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    if request.method == 'POST':
        form_image = request.files.get('target_image')
        filename = None
        if form_image and form_image.filename != '':
            filename = secure_filename(form_image.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            form_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        arrows_per_end = int(request.form.get('arrows_per_end') or 6)
        clicked_arrows = json.loads(request.form.get('arrow_data') or '[]')

        # Canvas pixels -> mm from the target centre; the score is recomputed on the server
        arrow_rows = []
        for i, a in enumerate(clicked_arrows):
            x_mm = (float(a['x']) - CANVAS_CENTER_PX) * MM_PER_PX
            y_mm = (CANVAS_CENTER_PX - float(a['y'])) * MM_PER_PX
            arrow_rows.append(Arrow(end_no=i // arrows_per_end + 1, arrow_no=i % arrows_per_end + 1,
                                    x_mm=round(x_mm, 1), y_mm=round(y_mm, 1), score=ring_score(x_mm, y_mm)))
        arrows_shot_count = len(arrow_rows)

        new_score = Score(
            date=date.fromisoformat(request.form.get('date')), distance=request.form.get('distance'),
            arrows_shot=arrows_shot_count, total_score=sum(a.score for a in arrow_rows),
            arrows_per_end=arrows_per_end,
            image_file=filename, weather_temp=request.form.get('weather_temp'),
            wind_speed=request.form.get('wind_speed'), wind_direction=request.form.get('wind_direction'),
            fatigue_level=request.form.get('fatigue_level'), focus_level=request.form.get('focus_level'),
            archer_id=session['user_id'], arrows=arrow_rows
        )
        db.session.add(new_score)
        archer = Archer.query.get(session['user_id'])
        archer.current_string_arrows = (archer.current_string_arrows or 0) + arrows_shot_count
        db.session.commit()
        try:
            refresh_forecast(archer.id)
        except Exception as e:  # a failed forecast must never lose the saved session
            app.logger.warning('Forecast refresh failed: %s', e)
            db.session.rollback()
        return redirect(url_for('my_trainings_page'))
    return render_template('add_score.html')


@app.route('/reset_string_lifespan')
def reset_string_lifespan():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    archer = Archer.query.get(session['user_id'])
    archer.current_string_arrows = 0
    db.session.commit()
    return redirect(url_for('dashboard_page'))


@app.route('/export_csv')
def export_csv():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'Distance', 'Arrows', 'Total Score', 'Fatigue', 'Focus'])
    scores = Score.query.filter_by(archer_id=session['user_id']).all()
    for s in scores:
        cw.writerow([s.date, s.distance, s.arrows_shot, s.total_score, s.fatigue_level, s.focus_level])
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=training_history.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@app.route('/add_sight_mark', methods=['GET', 'POST'])
def add_sight_mark_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    if request.method == 'POST':
        form_image = request.files.get('sight_image')
        filename = None
        if form_image and form_image.filename != '':
            filename = secure_filename(form_image.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            form_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        db.session.add(SightMark(date_recorded=request.form.get('date'), distance=request.form.get('distance'),
                                 setting=request.form.get('setting'), image_file=filename,
                                 archer_id=session['user_id']))
        db.session.commit()
        return redirect(url_for('dashboard_page'))
    return render_template('add_sight_mark.html')


@app.route('/add_tuning', methods=['GET', 'POST'])
def add_tuning_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    if request.method == 'POST':
        new_tuning = BowTuning(
            date_recorded=request.form.get('date'), bow_name=request.form.get('bow_name'),
            brace_height=request.form.get('brace_height'), tiller_upper=request.form.get('tiller_upper'),
            tiller_lower=request.form.get('tiller_lower'), nocking_point=request.form.get('nocking_point'),
            draw_weight=request.form.get('draw_weight'), notes=request.form.get('notes'),
            archer_id=session['user_id']
        )
        db.session.add(new_tuning)
        db.session.commit()
        return redirect(url_for('tuning_log_page'))
    return render_template('add_tuning.html')


@app.route('/tuning_log')
def tuning_log_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    logs = BowTuning.query.filter_by(archer_id=session['user_id']).order_by(
        BowTuning.date_recorded.desc(), BowTuning.id.desc()).all()
    return render_template('tuning_log.html', logs=logs)


@app.route('/my_trainings')
def my_trainings_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    scores = Score.query.filter_by(archer_id=session['user_id']).order_by(Score.id.desc()).all()
    return render_template('my_trainings.html', scores=scores)


@app.route('/view_training/<int:score_id>')
def view_training_page(score_id):
    if 'user_id' not in session: return redirect(url_for('login_page'))
    score = Score.query.get_or_404(score_id)
    if score.archer_id == session['user_id'] or session.get('is_coach'):
        # mm from the centre -> canvas pixels for drawing
        arrows = [dict(x=CANVAS_CENTER_PX + a.x_mm / MM_PER_PX, y=CANVAS_CENTER_PX - a.y_mm / MM_PER_PX,
                       score=a.score, end=a.end_no) for a in score.arrows]
        return render_template('view_training.html', score=score, arrows=arrows)
    return redirect(url_for('home_page'))



# --- FORECASTING ---
# PyTorch is only imported when a forecast is needed, so ordinary pages stay fast.
_forecast_model = None
_training = dict(running=False, started=None, finished=None, error=None, log=[])


def get_forecast_model():
    global _forecast_model
    if _forecast_model is None:
        import forecasting
        _forecast_model = forecasting.load_model()
    return _forecast_model


def reset_forecast_model():
    global _forecast_model
    _forecast_model = None


def refresh_forecast(archer_id):
    """Recompute one archer's 30-day forecast with the saved model (no retraining)."""
    import forecasting
    rows = forecasting.forecast(get_forecast_model(), forecasting.load_features(db.engine, archer_id))
    # Forget already-loaded forecast objects so the bulk delete cannot clash with the new rows
    for obj in [o for o in db.session.identity_map.values() if isinstance(o, Forecast)]: db.session.expunge(obj)
    Forecast.query.filter_by(archer_id=archer_id).delete(synchronize_session=False)
    if rows:
        now = datetime.now()
        db.session.add_all(Forecast(archer_id=archer_id, created_at=now, **r) for r in rows)
    db.session.commit()
    return bool(rows)


def run_training_in_background():
    def work():
        try:
            import train_forecast
            train_forecast.run(log=_training['log'].append)
        except Exception as e:
            _training['error'] = str(e)
        finally:
            _training['running'], _training['finished'] = False, datetime.now().strftime('%Y-%m-%d %H:%M')

    _training.update(running=True, started=datetime.now().strftime('%Y-%m-%d %H:%M'), finished=None, error=None, log=[])
    threading.Thread(target=work, daemon=True).start()


PERIODS = {'all': None, '90': 90, '30': 30}


def load_period_data(archer_id, period):
    all_sessions = analytics.load_sessions(db.engine, archer_id)
    arrows = analytics.load_arrows(db.engine, archer_id)
    plans = analytics.load_plans(db.engine, archer_id)
    sessions, arrows = analytics.filter_period(all_sessions, arrows, PERIODS[period])
    if PERIODS[period] and not sessions.empty:
        plans = plans[plans['date_assigned'] >= sessions['date'].min()]
    return sessions, arrows, plans, all_sessions


def render_analytics(archer, period, coach_view):
    if period not in PERIODS: period = 'all'
    sessions, arrows, plans, all_sessions = load_period_data(archer.id, period)
    report = analytics.full_report(sessions, arrows, plans) if len(sessions) >= 3 and len(arrows) else None
    if report:
        # "Last 30 days vs previous 30" always looks at the full history, whatever the period filter
        report['summary']['change_30'] = analytics.change_last_30(all_sessions)
        rows = Forecast.query.filter_by(archer_id=archer.id).order_by(Forecast.date).all()
        if rows:
            recent = all_sessions['avg'].tail(30).mean()
            report['forecast'] = dict(
                dates=[r.date.isoformat() for r in rows], predicted=[r.predicted for r in rows],
                lower=[r.lower for r in rows], upper=[r.upper for r in rows],
                month_mean=round(sum(r.predicted for r in rows) / len(rows), 2),
                change=round(sum(r.predicted for r in rows) / len(rows) - recent, 2) + 0.0,  # + 0.0 turns -0.0 into 0.0
                created_at=rows[0].created_at.strftime('%Y-%m-%d %H:%M'))
        report['plain'] = insights.plain_report(report, sessions)
    return render_template('analytics.html', archer=archer, report=report, period=period, coach_view=coach_view)


@app.route('/analytics')
def analytics_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    if session.get('is_coach'): return redirect(url_for('compare_page'))
    return render_analytics(db.session.get(Archer, session['user_id']), request.args.get('period', 'all'), False)


@app.route('/analytics/<int:athlete_id>')
def athlete_analytics_page(athlete_id):
    if 'user_id' not in session or not session.get('is_coach'): return redirect(url_for('home_page'))
    athlete = Archer.query.get_or_404(athlete_id)
    if athlete.is_coach: return redirect(url_for('coach_dashboard_page'))
    return render_analytics(athlete, request.args.get('period', 'all'), True)


@app.route('/coach/compare')
def compare_page():
    if 'user_id' not in session or not session.get('is_coach'): return redirect(url_for('home_page'))
    period = request.args.get('period', 'all')
    if period not in PERIODS: period = 'all'
    athletes = []
    for athlete in Archer.query.filter_by(is_coach=False).order_by(Archer.id).all():
        sessions, arrows, plans, _ = load_period_data(athlete.id, period)
        if len(sessions) < 3 or arrows.empty: continue
        plan_stats = analytics.plan_effect(plans, sessions)
        summary, fat = analytics.summary(sessions, arrows), analytics.in_session_fatigue(arrows)
        athletes.append(dict(id=athlete.id, name=athlete.first_name or athlete.username, summary=summary,
                             ma30=analytics.progress(sessions)['ma30'],
                             dates=[d.date().isoformat() for d in sessions['date']],
                             fatigue_drop=fat['drop'],
                             plan_rate=plan_stats['rate'] if plan_stats else None,
                             improvement=analytics.first_last_change(sessions),
                             trend=insights.trend(summary['recent_trend_per_30']),
                             consistency=insights.consistency(summary['day_to_day']),
                             fatigue=insights.fatigue(fat['first_half'], fat['second_half'])))
    return render_template('compare.html', athletes=athletes, period=period)


@app.route('/coach/model')
def model_page():
    if 'user_id' not in session or not session.get('is_coach'): return redirect(url_for('home_page'))
    import forecasting
    evaluation = forecasting.load_evaluation()
    return render_template('model.html', evaluation=evaluation, training=_training,
                           reliability=insights.model_reliability(evaluation))


@app.route('/coach/model/retrain', methods=['POST'])
def retrain_model():
    if 'user_id' not in session or not session.get('is_coach'): return redirect(url_for('home_page'))
    if not _training['running']: run_training_in_background()
    return redirect(url_for('model_page'))


@app.route('/dashboard')
def dashboard_page():
    if session.get('is_coach'): return redirect(url_for('coach_dashboard_page'))
    if 'user_id' in session:
        archer = Archer.query.get(session['user_id'])
        scores = Score.query.filter_by(archer_id=session['user_id']).all()
        return render_template('dashboard.html', username=session['username'], archer=archer,
                               scores=scores,
                               sight_marks=SightMark.query.filter_by(archer_id=session['user_id']).order_by(
                                   SightMark.date_recorded.desc(), SightMark.id.desc()).all(),
                               plans=TrainingPlan.query.filter_by(athlete_id=session['user_id']).order_by(
                                   TrainingPlan.date_assigned.desc(), TrainingPlan.id.desc()).all(),
                               total_sessions=len(scores),
                               total_arrows_ever=sum(s.arrows_shot for s in scores),
                               perfect_10s=Arrow.query.join(Score).filter(Score.archer_id == session['user_id'],
                                                                          Arrow.score == 10).count())
    return redirect(url_for('login_page'))


@app.route('/logout')
def logout_page():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_coach', None)
    return redirect(url_for('home_page'))


if __name__ == '__main__':
    app.run(debug=True)