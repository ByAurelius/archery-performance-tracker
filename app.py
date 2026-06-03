from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash

app = Flask(__name__)

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

# Create the database tables
with app.app_context():
    db.create_all()

@app.route('/')
def home_page():
    return render_template('index.html')

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        # Get data from the HTML form
        form_username = request.form.get('username')
        form_email = request.form.get('email')
        form_password = request.form.get('password')

        # Hash the password for security
        hashed_password = generate_password_hash(form_password)

        # Create new archer and save to database
        new_archer = Archer(username=form_username, email=form_email, password=hashed_password)
        db.session.add(new_archer)
        db.session.commit()

        # Redirect to home page after successful registration
        return redirect(url_for('home_page'))

    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)