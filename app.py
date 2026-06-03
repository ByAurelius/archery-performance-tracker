import os
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
    scores = db.relationship('Score', backref='shooter', lazy=True)
    sight_marks = db.relationship('SightMark', backref='shooter', lazy=True)
    # Yeni Iliski: Sporcunun antrenman programlari
    training_plans = db.relationship('TrainingPlan', backref='receiver', lazy=True)


class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20), nullable=False)
    distance = db.Column(db.Integer, nullable=False)
    arrows_shot = db.Column(db.Integer, nullable=False)
    total_score = db.Column(db.Integer, nullable=False)
    image_file = db.Column(db.String(255), nullable=True)
    archer_id = db.Column(db.Integer, db.ForeignKey('archer.id'), nullable=False)


class SightMark(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_recorded = db.Column(db.String(20), nullable=False)
    distance = db.Column(db.Integer, nullable=False)
    setting = db.Column(db.String(50), nullable=False)
    archer_id = db.Column(db.Integer, db.ForeignKey('archer.id'), nullable=False)


# YENI TABLO: Antrenman Programlari (Training Plans)
class TrainingPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)  # Orn: Monday Technical Training
    description = db.Column(db.Text, nullable=False)  # Orn: 70m 90 arrows + physical conditioning
    date_assigned = db.Column(db.String(20), nullable=False)
    is_completed = db.Column(db.Boolean, default=False)  # Odev bitti mi?
    athlete_id = db.Column(db.Integer, db.ForeignKey('archer.id'), nullable=False)


with app.app_context():
    db.create_all()


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        form_username = request.form.get('username')
        form_email = request.form.get('email')
        form_password = request.form.get('password')
        form_is_coach = request.form.get('is_coach') == 'on'

        hashed_password = generate_password_hash(form_password)
        new_archer = Archer(username=form_username, email=form_email, password=hashed_password, is_coach=form_is_coach)
        db.session.add(new_archer)
        db.session.commit()
        return redirect(url_for('login_page'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        form_email = request.form.get('email')
        form_password = request.form.get('password')
        archer = Archer.query.filter_by(email=form_email).first()

        if archer and check_password_hash(archer.password, form_password):
            session['user_id'] = archer.id
            session['username'] = archer.username
            session['is_coach'] = archer.is_coach
            if archer.is_coach:
                return redirect(url_for('coach_dashboard_page'))
            else:
                return redirect(url_for('dashboard_page'))
        else:
            return "<h1>Invalid Email or Password!</h1>"
    return render_template('login.html')


@app.route('/coach_dashboard')
def coach_dashboard_page():
    if 'user_id' in session and session.get('is_coach'):
        athletes = Archer.query.filter_by(is_coach=False).all()
        return render_template('coach_dashboard.html', username=session['username'], athletes=athletes)
    else:
        return redirect(url_for('home_page'))


@app.route('/athlete/<int:athlete_id>')
def athlete_stats_page(athlete_id):
    if 'user_id' not in session or not session.get('is_coach'):
        return redirect(url_for('home_page'))

    athlete = Archer.query.get_or_404(athlete_id)
    if athlete.is_coach:
        return redirect(url_for('coach_dashboard_page'))

    athlete_scores = Score.query.filter_by(archer_id=athlete.id).all()
    athlete_sight_marks = SightMark.query.filter_by(archer_id=athlete.id).all()
    # Sporcuya ait odevleri de sayfaya gonderiyoruz
    athlete_plans = TrainingPlan.query.filter_by(athlete_id=athlete.id).all()

    return render_template('athlete_stats.html', athlete=athlete, scores=athlete_scores,
                           sight_marks=athlete_sight_marks, plans=athlete_plans)


# YENI ROUTE: Antrenorun Odev Atamasi
@app.route('/assign_plan/<int:athlete_id>', methods=['POST'])
def assign_plan(athlete_id):
    if 'user_id' not in session or not session.get('is_coach'):
        return redirect(url_for('home_page'))

    form_title = request.form.get('title')
    form_desc = request.form.get('description')
    form_date = request.form.get('date_assigned')

    new_plan = TrainingPlan(title=form_title, description=form_desc, date_assigned=form_date, athlete_id=athlete_id)
    db.session.add(new_plan)
    db.session.commit()

    return redirect(url_for('athlete_stats_page', athlete_id=athlete_id))


# YENI ROUTE: Sporcunun Odevi "Tamamlandi" Olarak Isaretlemesi
@app.route('/complete_plan/<int:plan_id>')
def complete_plan(plan_id):
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    plan = TrainingPlan.query.get_or_404(plan_id)

    # Guvenlik kontrolu: Odev gerçekten bu sporcuya mi ait?
    if plan.athlete_id == session['user_id']:
        plan.is_completed = True
        db.session.commit()

    return redirect(url_for('dashboard_page'))


@app.route('/add_score', methods=['GET', 'POST'])
def add_score_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    if request.method == 'POST':
        form_date = request.form.get('date')
        form_distance = request.form.get('distance')
        form_arrows = request.form.get('arrows_shot')
        form_score = request.form.get('total_score')
        form_image = request.files.get('target_image')
        filename = None

        if form_image and form_image.filename != '':
            filename = secure_filename(form_image.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            form_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        new_score = Score(date=form_date, distance=form_distance, arrows_shot=form_arrows,
                          total_score=form_score, image_file=filename, archer_id=session['user_id'])
        db.session.add(new_score)
        db.session.commit()
        return redirect(url_for('dashboard_page'))
    return render_template('add_score.html')


@app.route('/add_sight_mark', methods=['GET', 'POST'])
def add_sight_mark_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    if request.method == 'POST':
        form_date = request.form.get('date')
        form_distance = request.form.get('distance')
        form_setting = request.form.get('setting')

        new_mark = SightMark(date_recorded=form_date, distance=form_distance, setting=form_setting,
                             archer_id=session['user_id'])
        db.session.add(new_mark)
        db.session.commit()
        return redirect(url_for('dashboard_page'))
    return render_template('add_sight_mark.html')


@app.route('/dashboard')
def dashboard_page():
    if session.get('is_coach'): return redirect(url_for('coach_dashboard_page'))

    if 'user_id' in session:
        user_scores = Score.query.filter_by(archer_id=session['user_id']).all()
        user_sight_marks = SightMark.query.filter_by(archer_id=session['user_id']).all()
        # Sporcunun kendi ekraninda odevlerini listelemesi icin veriyi cekiyoruz
        user_plans = TrainingPlan.query.filter_by(athlete_id=session['user_id']).all()
        return render_template('dashboard.html', username=session['username'], scores=user_scores,
                               sight_marks=user_sight_marks, plans=user_plans)
    else:
        return redirect(url_for('login_page'))


@app.route('/logout')
def logout_page():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_coach', None)
    return redirect(url_for('home_page'))


if __name__ == '__main__':
    app.run(debug=True)