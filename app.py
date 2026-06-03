from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

app.secret_key = 'archery_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///archery.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Model: Archer (Sporcu Tablosu)
class Archer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    # Relationship: Bir sporcunun birden fazla skoru olabilir
    scores = db.relationship('Score', backref='shooter', lazy=True)

# Database Model: Score (Skor/Antrenman Tablosu)
class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20), nullable=False)
    distance = db.Column(db.Integer, nullable=False) # Kac metreden atildi
    arrows_shot = db.Column(db.Integer, nullable=False) # Kac ok atildi
    total_score = db.Column(db.Integer, nullable=False) # Toplam Puan
    # Foreign Key: Bu skor hangi sporcuya ait?
    archer_id = db.Column(db.Integer, db.ForeignKey('archer.id'), nullable=False)

with app.app_context():
    db.create_all() # Yeni tabloyu veritabanina ekler

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        form_username = request.form.get('username')
        form_email = request.form.get('email')
        form_password = request.form.get('password')

        hashed_password = generate_password_hash(form_password)
        new_archer = Archer(username=form_username, email=form_email, password=hashed_password)
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
            return redirect(url_for('dashboard_page'))
        else:
            return "<h1>Invalid Email or Password!</h1>"

    return render_template('login.html')

# Add Score Route (Skor Ekleme Sayfasi)
@app.route('/add_score', methods=['GET', 'POST'])
def add_score_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    if request.method == 'POST':
        form_date = request.form.get('date')
        form_distance = request.form.get('distance')
        form_arrows = request.form.get('arrows_shot')
        form_score = request.form.get('total_score')

        # Veritabanina yeni skoru kaydet
        new_score = Score(
            date=form_date,
            distance=form_distance,
            arrows_shot=form_arrows,
            total_score=form_score,
            archer_id=session['user_id']
        )
        db.session.add(new_score)
        db.session.commit()
        return redirect(url_for('dashboard_page'))

    return render_template('add_score.html')

# Dashboard Route (Guncellendi: Artik skorlari da veritabanindan cekiyor)
@app.route('/dashboard')
def dashboard_page():
    if 'user_id' in session:
        # Sisteme giren kisinin id'sine ait skorlari bul
        user_scores = Score.query.filter_by(archer_id=session['user_id']).all()
        return render_template('dashboard.html', username=session['username'], scores=user_scores)
    else:
        return redirect(url_for('login_page'))

@app.route('/logout')
def logout_page():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('home_page'))

if __name__ == '__main__':
    app.run(debug=True)