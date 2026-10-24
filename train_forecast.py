"""Trains and evaluates the LSTM forecaster, saves it, and refreshes every archer's forecast.

Usage:  python train_forecast.py

Archers with at least VAL_END + HORIZON sessions take part in training and evaluation.
Afterwards every archer with at least INPUT_DAYS sessions gets a fresh 30-day forecast.
Results are written to models/evaluation.json and the model to models/lstm_forecast.pt.
"""
import time

import forecasting
from app import app, db, Archer, Score, refresh_forecast, reset_forecast_model


def run(log=print):
    started = time.time()
    with app.app_context():
        archers = []
        for a in Archer.query.filter_by(is_coach=False).order_by(Archer.id):
            if Score.query.filter_by(archer_id=a.id).count() < forecasting.VAL_END + forecasting.HORIZON: continue
            archers.append(dict(name=a.first_name or a.username, username=a.username,
                                df=forecasting.load_features(db.engine, a.id)))
        if not archers:
            log('No archer has enough sessions to train on.')
            return None
        log(f"Archers: {', '.join(a['name'] for a in archers)}")
        results, model = forecasting.evaluate(archers, log=log)
        results['trained_at'] = time.strftime('%Y-%m-%d %H:%M')
        results['duration_s'] = round(time.time() - started, 1)
        forecasting.save(model, results)
        reset_forecast_model()

        updated = 0
        for a in Archer.query.filter_by(is_coach=False):
            if refresh_forecast(a.id): updated += 1
        log(f'Saved model and evaluation; refreshed {updated} forecasts in {time.time() - started:.0f} s.')
        return results


if __name__ == '__main__':
    run()
