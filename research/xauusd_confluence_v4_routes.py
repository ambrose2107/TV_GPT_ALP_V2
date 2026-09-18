from flask import Blueprint, render_template, request, jsonify, session, send_file
import logging
import time
import math
import os
import json
import csv
from pathlib import Path
import numpy as np
from research.xauusd_confluence_v4 import V4Config, load_data, backtest, optimize

logger = logging.getLogger("xauusd_confluence_v4")

xauusd_confluence_v4_bp = Blueprint('xauusd_confluence_v4', __name__)


def _storage_dir():
    root = os.environ.get('XAUUSD_V4_STORAGE_DIR', 'data/xauusd_v4_runs')
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_run_files(symbol, mode, payload):
    stamp = time.strftime('%Y%m%d_%H%M%S')
    safe_symbol = ''.join(ch for ch in symbol if ch.isalnum() or ch in ('_', '-')) or 'GLD'
    folder = _storage_dir()
    prefix = folder / f'{stamp}_{safe_symbol}_{mode}'
    json_path = prefix.with_suffix('.json')
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    files = {'json': str(json_path)}
    if mode == 'backtest':
        rows = payload.get('trades', [])
        p = prefix.with_name(prefix.name + '_trades.csv')
        if rows:
            with p.open('w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            p.write_text('', encoding='utf-8')
        files['trades'] = str(p)
        rows = payload.get('signals', [])
        p = prefix.with_name(prefix.name + '_signals.csv')
        if rows:
            with p.open('w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            p.write_text('', encoding='utf-8')
        files['signals'] = str(p)
    else:
        rows = payload.get('optimizer', {}).get('results', [])
        p = prefix.with_name(prefix.name + '_optimizer.csv')
        if rows:
            with p.open('w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            p.write_text('', encoding='utf-8')
        files['optimizer_csv'] = str(p)
    return files



def _auth():
    if not session.get('logged_in'):
        return jsonify({'error':'Unauthorized'}), 401
    return None

def _json_safe(value):
    """Convert numpy/non-finite values to standard JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


@xauusd_confluence_v4_bp.route('/api/xauusd-confluence-v4/download/<path:filename>')
def download(filename):
    e = _auth()
    if e: return e
    root = _storage_dir().resolve()
    target = (root / filename).resolve()
    if root not in target.parents:
        return jsonify({'error': 'Invalid file path'}), 400
    if not target.exists() or not target.is_file():
        return jsonify({'error': 'File not found'}), 404
    return send_file(target, as_attachment=True, download_name=target.name)

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
        logger.info('[RUN] start symbol=%s source=%s n_bars=%d min_score=%s min_rr=%s risk_pct=%s',
                    symbol, data_source, n_bars, cfg.min_score, cfg.min_rr, cfg.risk_pct)

        stage='load_data'
        t=time.perf_counter()
        data=load_data(use_live=True,n_bars=n_bars,symbol=symbol,data_source=data_source)
        logger.info('[RUN] load_data complete bars=%d elapsed=%.2fs',
                    len(data['m5']), time.perf_counter()-t)

        stage='backtest'
        t=time.perf_counter()
        initial_equity=float(body.get('initial_equity',10000))
        if str(body.get('mode','backtest')).lower() == 'optimize':
            stage='optimize'
            opt=optimize(data,cfg,initial_equity=initial_equity,min_trades=int(body.get('min_trades',5)))
            response={'symbol':symbol,'data_source':data_source,'bars':len(data['m5']),
                'data_start':str(data['m5'].index.min()),'data_end':str(data['m5'].index.max()),
                'baseline_config':cfg.__dict__.copy(),'optimizer':opt}
            if opt.get('results'):
                best=opt['results'][0]
                response['best_params']={k:best[k] for k in ('min_score','min_rr','sl_atr','cooldown_bars')}
            safe=_json_safe(response)
            files=_save_run_files(symbol,'optimize',safe)
            safe['storage_files']={k:Path(v).name for k,v in files.items()}
            safe['storage_downloads']={k:f'/api/xauusd-confluence-v4/download/{Path(v).name}' for k,v in files.items()}
            safe['storage_dir']=str(_storage_dir())
            logger.info('[RUN] optimizer complete tested=%d eligible=%d elapsed=%.2fs saved=%s', opt['tested'], opt['eligible'], time.perf_counter()-t, files)
            return jsonify(safe)
        result=backtest(data,cfg,initial_equity=initial_equity)
        logger.info('[RUN] backtest complete trades=%d elapsed=%.2fs metrics=%s',
                    len(result['trades']), time.perf_counter()-t, result['metrics'])

        stage='serialize'
        trades=result['trades'].tail(200).copy()
        for col in ('entry_time','exit_time'):
            if col in trades:
                trades[col]=trades[col].astype(str)

        signals=result['signals'].tail(1000)[
            ['signal','score','sl','tp1','tp2','tp3','reason']
        ].reset_index().rename(columns={'index':'time'})

        response={
            'symbol':symbol,
            'data_source':data_source,
            'bars':len(data['m5']),
            'data_start':str(data['m5'].index.min()),
            'data_end':str(data['m5'].index.max()),
            'metrics':result['metrics'],
            'trades':trades.to_dict('records'),
            'signals':signals.to_dict('records'),
            'signal_diag':result['signals'].attrs.get('signal_diag',{})
        }
        safe_response=_json_safe(response)
        files=_save_run_files(symbol,'backtest',safe_response)
        safe_response['storage_files']={k:Path(v).name for k,v in files.items()}
        safe_response['storage_downloads']={k:f'/api/xauusd-confluence-v4/download/{Path(v).name}' for k,v in files.items()}
        safe_response['storage_dir']=str(_storage_dir())
        logger.info('[RUN] success total_elapsed=%.2fs payload_trades=%d payload_signals=%d saved=%s',
                    time.perf_counter()-started,len(safe_response['trades']),len(safe_response['signals']),files)
        return jsonify(safe_response)
    except Exception as ex:
        logger.exception('[RUN] failed stage=%s elapsed=%.2fs symbol=%s source=%s',
                         stage, time.perf_counter()-started,
                         body.get('symbol','GLD'), body.get('data_source','alpaca'))
        return jsonify({'error':str(ex),'type':type(ex).__name__,'stage':stage}),400
