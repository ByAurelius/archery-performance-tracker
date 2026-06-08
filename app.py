import os
import json
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.secret_key = 'archery_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///archery.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

db = SQLAlchemy(app)


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
    scores = db.relationship('Score', backref='shooter', lazy=True)
    sight_marks = db.relationship('SightMark', backref='shooter', lazy=True)
    training_plans = db.relationship('TrainingPlan', backref='receiver', lazy=True)
    tuning_logs = db.relationship('BowTuning', backref='shooter', lazy=True)


class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20), nullable=False)
    distance = db.Column(db.Integer, nullable=False)
    arrows_shot = db.Column(db.Integer, nullable=False)
    total_score = db.Column(db.Integer, nullable=False)
    image_file = db.Column(db.String(255), nullable=True)
    arrow_data = db.Column(db.Text, nullable=True)
    arrows_per_end = db.Column(db.Integer, default=6)
    weather_temp = db.Column(db.String(20), nullable=True)
    wind_speed = db.Column(db.String(20), nullable=True)
    wind_direction = db.Column(db.String(20), nullable=True)

    # YENI SUTUNLAR: Fiziksel ve Mental Durum (1-10)
    fatigue_level = db.Column(db.Integer, nullable=True)
    focus_level = db.Column(db.Integer, nullable=True)

    archer_id = db.Column(db.Integer, db.ForeignKey('archer.id'), nullable=False)


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


with app.app_context():
    db.create_all()


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
                           sight_marks=SightMark.query.filter_by(archer_id=athlete.id).all(),
                           plans=TrainingPlan.query.filter_by(athlete_id=athlete.id).all(),
                           tuning_logs=BowTuning.query.filter_by(archer_id=athlete.id).order_by(
                               BowTuning.id.desc()).all())


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

        new_score = Score(
            date=request.form.get('date'), distance=request.form.get('distance'),
            arrows_shot=request.form.get('arrows_shot'), total_score=request.form.get('total_score'),
            arrow_data=request.form.get('arrow_data'), arrows_per_end=request.form.get('arrows_per_end'),
            image_file=filename, weather_temp=request.form.get('weather_temp'),
            wind_speed=request.form.get('wind_speed'), wind_direction=request.form.get('wind_direction'),
            fatigue_level=request.form.get('fatigue_level'), focus_level=request.form.get('focus_level'),
            archer_id=session['user_id']
        )
        db.session.add(new_score)
        db.session.commit()
        return redirect(url_for('my_trainings_page'))
    return render_template('add_score.html')


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
    logs = BowTuning.query.filter_by(archer_id=session['user_id']).order_by(BowTuning.id.desc()).all()
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
        return render_template('view_training.html', score=score)
    return redirect(url_for('home_page'))


@app.route('/dashboard')
def dashboard_page():
    if session.get('is_coach'): return redirect(url_for('coach_dashboard_page'))
    if 'user_id' in session:
        return render_template('dashboard.html', username=session['username'],
                               scores=Score.query.filter_by(archer_id=session['user_id']).all(),
                               sight_marks=SightMark.query.filter_by(archer_id=session['user_id']).all(),
                               plans=TrainingPlan.query.filter_by(athlete_id=session['user_id']).all())
    return redirect(url_for('login_page'))


@app.route('/logout')
def logout_page():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_coach', None)
    return redirect(url_for('home_page'))


if __name__ == '__main__':
    app.run(debug=True)