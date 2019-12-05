import datetime
import math
import os
import pathlib
import re
import shutil
import sys
import munch
import pickle
import numpy as np
import pandas as pd
from numba import jit
import urllib.parse

SECONDS_IN_MS = 1000
MINUTES_IN_MS = 60 * SECONDS_IN_MS
HOURS_IN_MS = 60 * MINUTES_IN_MS
DAYS_IN_MS = 24 * HOURS_IN_MS
TIMESTAMP_MS_CHECK = 10 ** 10

def get_stocks(selected_index='^GDAXI'):
    enc_index = urllib.parse.quote(selected_index)
    print(f"shared> parsing stocks from web '{selected_index}' ...")
    index_stocks = pd.read_html(f'https://finance.yahoo.com/quote/{enc_index}/components?p={enc_index}')[0].Symbol.tolist()
    return sorted([t for t in index_stocks])

def get_benchmarks():
    print(f"shared> parsing benchmarks from web ...")
    indices = pd.read_html('https://finance.yahoo.com/world-indices')[0].Symbol.tolist()
    currencies = pd.read_html('https://finance.yahoo.com/currencies')[0].Symbol.tolist()
    commodities = pd.read_html('https://finance.yahoo.com/commodities')[0].Symbol.tolist()
    return sorted([i for i in (indices + currencies + commodities)])
    
def index_to_datetime(df, fmt='%Y'):
    if fmt=='%Q':
        df.index = df.index.astype(str).to_series().apply(lambda s: parse_datetime(s[-4:] + '-' + {'1Q': '03-31', '2Q': '06-30', '3Q': '09-30', '4Q': '12-31'}.get(s[:2])))
    else:
        df.index = pd.to_datetime(df.index.astype(str), format=fmt)
    return df

def parse_datetime(dt_str):
    l = len(dt_str)
    if l == 19:
        if dt_str[10] == ' ':
            return datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        else:
            return datetime.datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S')
    elif l > 19:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%f')
    elif l > 9:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d')
    elif l > 5:
        return datetime.datetime.strptime(dt_str, '%Y-%m')
    else:
        return datetime.datetime.strptime(dt_str, '%Y')

def get_todays_build_version(version='1.0.0'):
    today_dt = datetime.datetime.combine(datetime.datetime.now(), datetime.time.min)
    build = format_build_datetime(today_dt)
    return version, build, f'{version}_{build}'


def format_datetime(dt=datetime.datetime.now(), fmt='%Y-%m-%d %H:%M:%S'):
    if isinstance(dt, str):
        return dt
    else:
        return dt.strftime(fmt)


def format_build_datetime(dt=datetime.datetime.now()):
    if type(str) == str:
        return str.replace(':', '')
    else:
        return format_datetime(dt, '%Y%m%d%H%M%S')

def format_build_date(dt=datetime.datetime.now()):
    if type(str) == str:
        return str.replace(':', '').replace('-', '')
    else:
        return format_datetime(dt, '%Y%m%d')

def format_date(dt=datetime.datetime.now()):
    return format_datetime(dt, '%Y-%m-%d')


def chunks(l, n=200):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def chunks_idx(l, n=200):
    for i in range(0, len(l), n):
        yield i, i + n

def to_category_name(s):
    return re.sub('[^a-z0-9_]', '', s.strip().lower().replace(' ', '_'))

def to_category_names(df):
    if df is not None:
        df.rename(to_category_name, axis='columns', inplace=True)
        return df
    else:
        return df

def is_true(str):
    if type(str) == bool:
        return str
    else:
        return str.lower() in ['true', '1', 't', 'yes', 'y']


@jit(nopython=True)
def get_timestamp_ms(ts):
    if ts < TIMESTAMP_MS_CHECK:
        return ts * 1000
    else:
        return ts


@jit(nopython=True)
def get_timestamp_sec(ts):
    if ts > TIMESTAMP_MS_CHECK:
        return ts / 1000
    else:
        return ts

    
def get_config(selected_index='^GDAXI', overwrite=False):
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), verbose=True)
    
    # add current directory to sys.path
    p = pathlib.Path('.').resolve()
    current_dir = str(p)
    sys.path.append(current_dir)
    print(f"shared> current directory:{current_dir}")
    
    p = pathlib.Path(os.environ.get('CACHE_DIR', './cache/'))
    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    cache_dir = str(p.resolve())
    
    f_cfg = pathlib.Path(os.environ.get('CONFIG_FILE_PATH', 'config.json'))
    if not overwrite:        
        if f_cfg.is_file():    
            with open(f_cfg, 'r', encoding='utf-8') as fd:
                cfg = munch.Munch.fromJSON(fd.read())
                print(f'shared> read from config file: {cfg}')
                return cfg
    
    
    # feature configurations
    window_trading_days = [int(s.strip()) for s in os.environ['TRAIN_WINDOW_TRADING_DAYS'].split(',')]
    lag_trading_days = [int(s.strip()) for s in os.environ['TRAIN_LAG_TRADING_DAYS'].split(',')]
    
    # parse start and end dates
    end_dt_str = format_date() if len(os.environ['END_DT']) == 0 else os.environ['END_DT']
    end_dt = parse_datetime(end_dt_str)
    start_dt_str = None if len(os.environ['START_DT']) == 0 else os.environ['START_DT']
    if start_dt_str is not None:
        start_dt = parse_datetime(start_dt_str)
    else:
        weeks = int(os.environ.get('TRAIN_LAST_WEEKS', '12'))        
        start_dt = end_dt - datetime.timedelta(weeks=weeks)
        start_dt_str = format_date(start_dt)
    print(f"shared> training period: from '{start_dt_str}' to '{end_dt_str}'")
    
    max_window_trading_days = max(window_trading_days)
    conservative_download_days = math.ceil(7 + (max_window_trading_days / 5. * 7.))
    download_start_dt = start_dt - datetime.timedelta(days=conservative_download_days)
    download_start_dt_str = format_date(download_start_dt)
    cache_enabled = is_true(os.environ['CACHE_ENABLED'])

    print(f"shared> download period: from '{download_start_dt_str}' to '{end_dt_str}'")
    
    # read selected features
    if len(os.environ['BENCHMARKS']) > 0:
        benchmarks = sorted([s.upper().strip() for s in os.environ['BENCHMARKS'].split(',')])
    else:
        benchmarks = get_benchmarks()
    if len(os.environ['STOCKS']) > 0:
        stocks = sorted([s.upper().strip() for s in os.environ['STOCKS'].split(',')])
    else:
        stocks = get_stocks(selected_index)
    print(f"shared> benchmarks: '{benchmarks}'")
    print(f"shared> stocks: '{stocks}'")
    
    # read time horizons
    train_horizon_days = [int(s.strip()) for s in os.environ['TRAIN_HORIZON_DAYS'].split(',')]
    train_predict_days = [int(s.strip()) for s in os.environ['TRAIN_PREDICT_DAYS'].split(',')]
    train_horizon_cfg = list(zip(train_horizon_days, train_predict_days))
    
    # create config obj
    cfg = munch.munchify({
        'config_file_path': str(f_cfg.resolve()),
        'cache_dir': cache_dir,
        'cache_enabled': cache_enabled,
        'current_dir': current_dir,
        'start_dt_str': start_dt_str,
        'download_start_dt_str': download_start_dt_str,
        'end_dt_str': end_dt_str,
        'filter_start_dt_str': start_dt_str,
        'filter_end_dt_str': end_dt_str,
        'benchmarks': benchmarks,
        'stocks': stocks,
        'train_horizon_config': train_horizon_cfg,
        'train_window_trading_days': window_trading_days,
        'train_lag_trading_days': lag_trading_days
    })
    print(f'shared> read from .env: {cfg}')
    return cfg

def save_config(cfg):
    f_cfg = pathlib.Path(cfg.config_file_path)
    cfg_json = cfg.toJSON(indent=4, sort_keys=True)
    with open(f_cfg, 'w', encoding='utf-8') as fd:
        fd.write(cfg_json)
    print(f"shared> saved config to '{f_cfg.resolve()}'")

class ProviderException(Exception):
    pass


def filter_dates_generator(cfg, horizon_cfg):
    train_days = horizon_cfg[0]
    predict_days = horizon_cfg[1]
    start_dt = parse_datetime(cfg.start_dt_str)
    end_dt = parse_datetime(cfg.end_dt_str)
    train_start_dt = parse_datetime(cfg.start_dt_str)    
    while(True):
        train_end_dt = train_start_dt + datetime.timedelta(days=train_days)
        predict_start_dt = train_end_dt + datetime.timedelta(days=1)
        predict_end_dt = predict_start_dt + datetime.timedelta(days=predict_days-1)
        if predict_end_dt>end_dt:
            break
        else:
            yield(munch.munchify({
                'train_start_dt': train_start_dt,
                'train_end_dt': train_end_dt,
                'predict_start_dt': predict_start_dt,
                'predict_end_dt': predict_end_dt
            }))
        train_start_dt += datetime.timedelta(days=1)