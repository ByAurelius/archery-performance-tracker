"""Route tests for the Flask app, run against a temporary in-memory database.

Run with:  python -m pytest
"""
import json
import os

os.environ['DATABASE_URL'] = 'sqlite:///:memory:'  # must be set before the app is imported

import pytest

from app import app, db, Archer, Score, Arrow


@pytest.fixture
def client():
    app.config['TESTING'] = True
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
    client.post('/register', data={'username': 'coach', 'email': 'coach@test.com', 'password': 'pw-12345', 'is_coach': 'on'})
    client.post('/login', data={'email': 'coach@test.com', 'password': 'pw-12345'})
    r = client.get(f'/athlete/{athlete.id}?page=99')
    assert r.status_code == 302 and r.headers['Location'].endswith(f'/athlete/{athlete.id}?page=2')
    assert b'Showing 21' in client.get(f'/athlete/{athlete.id}?page=2').data
