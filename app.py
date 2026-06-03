from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Database Configuration (Veritabani Ayarlari)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///archery.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Model: Archer (Sporcu Tablosu)
class Archer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"Archer('{self.username}', '{self.email}')"

# Create the database file automatically (Veritabani dosyasini otomatik olusturur)
with app.app_context():
    db.create_all()

@app.route('/')
def home_page():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)