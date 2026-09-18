from flask import Blueprint, render_template, request, jsonify, session
import math
import numpy as np
from research.xauusd_confluence_v4 import load_data
from research.xauusd_pullback_v1 import PullbackConfig, backtest, optimize

xauusd_pullback_v1_bp = Blueprint('xauusd_pullback_v1', __name__)


def safe(v):
    if isinstance(v, dict):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    if isinstance(v, np.generic):
        return safe(v.item())
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


@xauusd_pullback_v1_bp.route('/xauusd-pullback-v1')
def page():
    if not session.get('logged_in'):
        from flask import redirect, url_for
        return redirect(url_for('dashboard.login'))
    return render_template('xauusd_pullback_v1.html')


@xauusd_pullback_v1_bp.route('/api/xauusd-pullback-v1/run', methods=['POST'])
def run():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    body = request.get_json(silent=True) or {}
    try:
        cfg = PullbackConfig(**{k: v for k, v in (body.get('config') or {}).items()
                                if k in PullbackConfig.__dataclass_fields__})
        bars = int(body.get('n_bars', 3900))
        symbol = str(body.get('symbol', 'GLD')).upper()
        data = load_data(use_live=True, n_bars=bars, symbol=symbol, data_source='alpaca')
        if str(body.get('mode', 'backtest')).lower() == 'optimize':
            return jsonify(safe({
                'symbol': symbol, 'bars': len(data['m5']),
                'optimizer': optimize(data['m5'], cfg, min_trades=int(body.get('min_trades', 20)))
            }))
        r = backtest(data['m5'], cfg)
        t = r['trades'].tail(200).copy()
        for col in ('entry_time', 'exit_time'):
            if col in t:
                t[col] = t[col].astype(str)
        return jsonify(safe({
            'symbol': symbol, 'bars': len(data['m5']),
            'data_start': str(data['m5'].index.min()), 'data_end': str(data['m5'].index.max()),
            'metrics': r['metrics'], 'trades': t.to_dict('records')
        }))
    except Exception as ex:
        return jsonify({'error': str(ex), 'type': type(ex).__name__}), 400
