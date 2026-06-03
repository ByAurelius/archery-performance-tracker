from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Security Key for Sessions (Oturum hafizasi icin guvenlik anahtari)
app.secret_key = 'archery_secret_key_123'

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///archery.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Model: Archer
class Archer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"Archer('{self.username}', '{self.email}')"

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

        hashed_password = generate_password_hash(form_password)

        new_archer = Archer(username=form_username, email=form_email, password=hashed_password)
        db.session.add(new_archer)
        db.session.commit()

        return redirect(url_for('login_page'))

    return render_template('register.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        form_email = request.form.get('email')
        form_password = request.form.get('password')

        # Find the user by email
        archer = Archer.query.filter_by(email=form_email).first()

        # Check if user exists and password is correct
        if archer and check_password_hash(archer.password, form_password):
            # Save user info in session memory
            session['user_id'] = archer.id
            session['username'] = archer.username
            return redirect(url_for('dashboard_page'))
        else:
            return "<h1>Invalid Email or Password!</h1>"

    return render_template('login.html')

# Dashboard Route
@app.route('/dashboard')
def dashboard_page():
    # Check if user is logged in
    if 'user_id' in session:
        return render_template('dashboard.html', username=session['username'])
    else:
        # If not logged in, send them back to login page
        return redirect(url_for('login_page'))

# Logout Route
@app.route('/logout')
def logout_page():
    # Clear the session memory
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('home_page'))

if __name__ == '__main__':
    app.run(debug=True)