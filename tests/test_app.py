"""Route tests for the Flask app, run against a temporary in-memory database.

Run with:  python -m pytest
"""
import io
import json
import os

os.environ['DATABASE_URL'] = 'sqlite:///:memory:'  # must be set before the app is imported

import pytest

from app import app, db, Archer, Score, Arrow, TrainingPlan

COACH_CODE = 'test-invite-code'


@pytest.fixture
def client(tmp_path, monkeypatch):
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # CSRF has its own test below
    app.config['UPLOAD_FOLDER'] = str(tmp_path)  # uploads never touch static/uploads
    monkeypatch.setenv('COACH_INVITE_CODE', COACH_CODE)
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


def register_and_login(client, name):
    client.post('/register', data={'username': name, 'email': f'{name}@test.com', 'password': 'pw-12345'})
    client.post('/login', data={'email': f'{name}@test.com', 'password': 'pw-12345'})
    return Archer.query.filter_by(username=name).one()


def add_session(client, n_arrows=6):
    arrows = [{'x': 200, 'y': 200}] * n_arrows  # all in the centre
    client.post('/add_score', data={'date': '2026-10-01', 'distance': 70, 'arrows_per_end': 6,
                                    'arrow_data': json.dumps(arrows)})
    return Score.query.order_by(Score.id.desc()).first()


def test_secret_key_is_not_hardcoded():
    assert app.secret_key != 'archery_secret_key_123'
    assert len(app.secret_key) >= 32


def test_delete_own_session_removes_it_with_its_arrows(client):
    archer = register_and_login(client, 'ayse')
    score = add_session(client)
    assert Arrow.query.filter_by(score_id=score.id).count() == 6
    assert db.session.get(Archer, archer.id).current_string_arrows == 6

    r = client.post(f'/delete_training/{score.id}', follow_redirects=True)
    assert r.status_code == 200 and b'was deleted' in r.data
    assert db.session.get(Score, score.id) is None
    assert Arrow.query.filter_by(score_id=score.id).count() == 0
    assert db.session.get(Archer, archer.id).current_string_arrows == 0


def test_cannot_delete_someone_elses_session(client):
    register_and_login(client, 'owner')
    score = add_session(client)
    client.get('/logout')
    register_and_login(client, 'intruder')

    r = client.post(f'/delete_training/{score.id}', follow_redirects=True)
    assert b'only delete your own' in r.data
    assert db.session.get(Score, score.id) is not None


def test_delete_needs_post_and_login(client):
    register_and_login(client, 'ali')
    score = add_session(client)
    assert client.get(f'/delete_training/{score.id}').status_code == 405  # a plain link cannot delete
    client.get('/logout')
    r = client.post(f'/delete_training/{score.id}')
    assert r.status_code == 302 and '/login' in r.headers['Location']
    assert db.session.get(Score, score.id) is not None


def test_training_history_is_paginated(client):
    register_and_login(client, 'zeynep')
    for _ in range(25): add_session(client)

    page1 = client.get('/my_trainings').data
    page2 = client.get('/my_trainings?page=2').data
    assert page1.count(b'View Scorecard') == 20       # SESSIONS_PER_PAGE
    assert page2.count(b'View Scorecard') == 5
    assert b'Showing 21' in page2 and b'of 25 sessions' in page2


def test_page_past_the_end_redirects_to_last_page(client):
    register_and_login(client, 'can')
    for _ in range(3): add_session(client)
    r = client.get('/my_trainings?page=99')
    assert r.status_code == 302 and r.headers['Location'].endswith('/my_trainings?page=1')


def test_coach_history_past_the_end_redirects_to_last_page(client):
    athlete = register_and_login(client, 'elif')
    for _ in range(25): add_session(client)
    client.get('/logout')
    client.post('/register', data={'username': 'coach', 'email': 'coach@test.com', 'password': 'pw-12345',
                                   'coach_code': COACH_CODE})
    client.post('/login', data={'email': 'coach@test.com', 'password': 'pw-12345'})
    r = client.get(f'/athlete/{athlete.id}?page=99')
    assert r.status_code == 302 and r.headers['Location'].endswith(f'/athlete/{athlete.id}?page=2')
    assert b'Showing 21' in client.get(f'/athlete/{athlete.id}?page=2').data


# --- Security fixes ---

def test_post_without_csrf_token_is_rejected(client):
    app.config['WTF_CSRF_ENABLED'] = True
    r = client.post('/login', data={'email': 'a@test.com', 'password': 'pw-12345'})
    assert r.status_code == 400 and b'expired' in r.data
    assert b'name="csrf_token"' in client.get('/login').data  # the real form carries the token


def test_data_changing_routes_refuse_get(client):
    archer = register_and_login(client, 'mert')
    db.session.add(TrainingPlan(title='Core', description='Plank', date_assigned='2026-10-01', athlete_id=archer.id))
    db.session.commit()
    plan = TrainingPlan.query.one()
    assert client.get(f'/complete_plan/{plan.id}').status_code == 405
    assert client.get('/reset_string_lifespan').status_code == 405
    client.post(f'/complete_plan/{plan.id}')
    assert db.session.get(TrainingPlan, plan.id).is_completed


def test_coach_account_needs_the_invite_code(client):
    r = client.post('/register', data={'username': 'fake', 'email': 'fake@test.com', 'password': 'pw-12345',
                                       'coach_code': 'wrong-code'})
    assert b'invite code is not correct' in r.data and Archer.query.filter_by(username='fake').first() is None
    client.post('/register', data={'username': 'real', 'email': 'real@test.com', 'password': 'pw-12345',
                                   'coach_code': COACH_CODE})
    assert Archer.query.filter_by(username='real').one().is_coach
    register_and_login(client, 'athlete')
    assert not Archer.query.filter_by(username='athlete').one().is_coach


def test_no_coach_sign_up_without_a_configured_code(client, monkeypatch):
    monkeypatch.setenv('COACH_INVITE_CODE', 'change-me')  # placeholder from .env.example
    client.post('/register', data={'username': 'x', 'email': 'x@test.com', 'password': 'pw-12345', 'coach_code': 'change-me'})
    assert Archer.query.filter_by(username='x').first() is None


def test_duplicate_email_shows_a_message_instead_of_crashing(client):
    register_and_login(client, 'ayse')
    client.get('/logout')
    r = client.post('/register', data={'username': 'ayse2', 'email': 'ayse@test.com', 'password': 'pw-12345'})
    assert r.status_code == 200 and b'already exists' in r.data
    assert b'value="ayse2"' in r.data  # the typed username is kept


def test_short_password_is_rejected(client):
    r = client.post('/register', data={'username': 'bob', 'email': 'bob@test.com', 'password': 'short'})
    assert b'at least 8 characters' in r.data and Archer.query.filter_by(username='bob').first() is None


def test_failed_login_shows_a_message(client):
    register_and_login(client, 'zehra')
    client.get('/logout')
    r = client.post('/login', data={'email': 'zehra@test.com', 'password': 'wrong-password'})
    assert r.status_code == 200 and b'Invalid email or password.' in r.data
    assert b'value="zehra@test.com"' in r.data


def test_upload_rejects_non_images_and_renames_images(client, tmp_path):
    register_and_login(client, 'deniz')
    r = client.post('/add_sight_mark', data={'date': '2026-10-01', 'distance': 70, 'setting': '4.85',
                                             'sight_image': (io.BytesIO(b'hello'), 'notes.txt')},
                    content_type='multipart/form-data')
    assert r.status_code == 400 and b'Only image files' in r.data and list(tmp_path.iterdir()) == []

    client.post('/add_sight_mark', data={'date': '2026-10-01', 'distance': 70, 'setting': '4.85',
                                         'sight_image': (io.BytesIO(b'fake-png'), 'my photo.PNG')},
                content_type='multipart/form-data')
    saved = [p.name for p in tmp_path.iterdir()]
    assert len(saved) == 1 and saved[0].endswith('.png') and saved[0] != 'my_photo.PNG'


def test_upload_larger_than_5_mb_is_rejected(client):
    register_and_login(client, 'efe')
    big = io.BytesIO(b'0' * (5 * 1024 * 1024 + 1))
    r = client.post('/add_sight_mark', data={'date': '2026-10-01', 'distance': 70, 'setting': '4.85',
                                             'sight_image': (big, 'big.png')}, content_type='multipart/form-data')
    assert r.status_code == 413 and b'File too large' in r.data
