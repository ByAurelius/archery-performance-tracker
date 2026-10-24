"""Model selection for the LSTM forecaster, using the validation set only.

Usage:  python tune_forecast.py

Trains one model per candidate setting on the training windows and reports the
validation MAE next to the 30-session moving average. The test set is never
touched here, so choosing the best row does not leak test information.
"""
import forecasting as F
from app import app, db, Archer

CANDIDATES = [  # (anchor days, hidden units, dropout)
    (7, 64, 0.2), (7, 32, 0.3), (30, 64, 0.2), (30, 32, 0.3),
]


def main():
    with app.app_context():
        base = [dict(name=a.first_name or a.username, username=a.username, df=F.load_features(db.engine, a.id))
                for a in Archer.query.filter_by(is_coach=False).order_by(Archer.id)
                if len(a.scores) >= F.VAL_END + F.HORIZON]
    default_anchor = F.ANCHOR_DAYS
    print(f"{'anchor':>6} {'hidden':>6} {'dropout':>7} {'epoch':>5} {'LSTM MAE':>9} {'MA30 MAE':>9} {'LSTM month':>10} {'MA30 month':>10}")
    for anchor, hidden, dropout in CANDIDATES:
        F.ANCHOR_DAYS = anchor
        archers = F.prepare([dict(a) for a in base])
        X, y = F.stack_split(archers, 'train')
        X_val, y_val = F.stack_split(archers, 'val')
        model, _, epoch = F.train_model(X, y, X_val, y_val, hidden=hidden, dropout=dropout)
        pairs = {}
        for a in archers:
            for name, p in F.test_errors(model, a, include_oracle=False, split='val')[0].items():
                pairs.setdefault(name, []).extend(p)
        s = F.summarise(pairs)
        lstm, ma = s['LSTM'], s['Moving average (last 30)']
        print(f'{anchor:>6} {hidden:>6} {dropout:>7} {epoch:>5} {lstm["mae"]:>9.4f} {ma["mae"]:>9.4f} '
              f'{lstm["month_error"]:>10.4f} {ma["month_error"]:>10.4f}')
    F.ANCHOR_DAYS = default_anchor


if __name__ == '__main__':
    main()
