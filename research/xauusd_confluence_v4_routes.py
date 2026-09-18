from flask import Blueprint, render_template, request, jsonify, session
import logging
import time
from research.xauusd_confluence_v4 import V4Config, load_data, backtest

logger = logging.getLogger("xauusd_confluence_v4")

xauusd_confluence_v4_bp = Blueprint('xauusd_confluence_v4', __name__)

def _auth():
    if not session.get('logged_in'):
        return jsonify({'error':'Unauthorized'}), 401
    return None

@xauusd_confluence_v4_bp.route('/xauusd-confluence-v4')
def page():
    if not session.get('logged_in'):
        from flask import redirect, url_for
        return redirect(url_for('dashboard.login'))
    return render_template('xauusd_confluence_v4.html')

@xauusd_confluence_v4_bp.route('/api/xauusd-confluence-v4/run', methods=['POST'])
def run():
    e=_auth()
    if e: return e
    body=request.get_json(silent=True) or {}
    started=time.perf_counter()
    stage='request'
    try:
        fields=V4Config.__dataclass_fields__.keys()
        cfg=V4Config(**{k:v for k,v in (body.get('config') or {}).items() if k in fields})
        symbol=str(body.get('symbol','GLD')).upper().strip()
        data_source=str(body.get('data_source','alpaca')).lower().strip()
        n_bars=int(body.get('n_bars',30000))
        logger.info('[RUN] start symbol=%s source=%s n_bars=%d min_score=%s risk_pct=%s',
                    symbol, data_source, n_bars, cfg.min_score, cfg.risk_pct)

        stage='load_data'
        t=time.perf_counter()
        data=load_data(use_live=True,n_bars=n_bars,symbol=symbol,data_source=data_source)
        logger.info('[RUN] load_data complete bars=%d elapsed=%.2fs',
                    len(data['m5']), time.perf_counter()-t)

        stage='backtest'
        t=time.perf_counter()
        result=backtest(data,cfg,initial_equity=float(body.get('initial_equity',10000)))
        logger.info('[RUN] backtest complete trades=%d elapsed=%.2fs metrics=%s',
                    len(result['trades']), time.perf_counter()-t, result['metrics'])

        stage='serialize'
        trades=result['trades'].tail(200).copy()
        for col in ('entry_time','exit_time'):
            if col in trades: trades[col]=trades[col].astype(str)
        signals=result['signals'].tail(1000)[['signal','score','sl','tp1','tp2','tp3','reason']].reset_index().rename(columns={'index':'time'})
        response={'symbol':symbol,'data_source':data_source,'bars':len(data['m5']),
                  'data_start':str(data['m5'].index.min()),'data_end':str(data['m5'].index.max()),
                  'metrics':result['metrics'],'trades':trades.to_dict('records'),
                  'signals':signals.to_dict('records')}
        logger.info('[RUN] success total_elapsed=%.2fs', time.perf_counter()-started)
        return jsonify(response)
    except Exception as ex:
        logger.exception('[RUN] failed stage=%s elapsed=%.2fs symbol=%s source=%s',
                         stage, time.perf_counter()-started,
                         body.get('symbol','GLD'), body.get('data_source','alpaca'))
        return jsonify({'error':str(ex),'type':type(ex).__name__,'stage':stage}),400
