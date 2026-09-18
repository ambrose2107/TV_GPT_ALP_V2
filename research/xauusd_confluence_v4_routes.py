from flask import Blueprint, render_template, request, jsonify, session
from research.xauusd_confluence_v4 import V4Config, load_data, backtest

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
    try:
        fields=V4Config.__dataclass_fields__.keys()
        cfg=V4Config(**{k:v for k,v in (body.get('config') or {}).items() if k in fields})
        symbol=str(body.get('symbol','GLD')).upper().strip()
        data_source=str(body.get('data_source','alpaca')).lower().strip()
        data=load_data(use_live=True,n_bars=int(body.get('n_bars',30000)),symbol=symbol,data_source=data_source)
        result=backtest(data,cfg,initial_equity=float(body.get('initial_equity',10000)))
        trades=result['trades'].tail(200).copy()
        for c in ('entry_time','exit_time'):
            if c in trades: trades[c]=trades[c].astype(str)
        return jsonify({'symbol':symbol,'data_source':data_source,'bars':len(data['m5']),'data_start':str(data['m5'].index.min()),'data_end':str(data['m5'].index.max()),'metrics':result['metrics'],'trades':trades.to_dict('records'),'signals':result['signals'].tail(1000)[['signal','score','sl','tp1','tp2','tp3','reason']].reset_index().rename(columns={'index':'time'}).to_dict('records')})
    except Exception as ex:
        import logging
        logging.getLogger(__name__).exception('XAUUSD V4 run failed')
        return jsonify({'error':str(ex),'type':type(ex).__name__}),400
