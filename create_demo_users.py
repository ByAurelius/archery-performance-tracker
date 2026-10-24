"""Creates demo accounts for local testing. Safe to run more than once.

Usage:  python create_demo_users.py
"""
from werkzeug.security import generate_password_hash
from app import app, db, Archer

DEMO_USERS = [
    # username, email, password, is_coach
    ('demo_athlete', 'athlete@demo.com', 'Athlete123!', False),
    ('demo_coach', 'coach@demo.com', 'Coach123!', True),
]

with app.app_context():
    for username, email, password, is_coach in DEMO_USERS:
        if Archer.query.filter_by(email=email).first():
            print(f'Already exists: {email}')
            continue
        db.session.add(Archer(username=username, email=email, password=generate_password_hash(password),
                              is_coach=is_coach))
        print(f'Created: {email}')
    db.session.commit()
