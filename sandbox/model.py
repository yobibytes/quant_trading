import collections
import os
import pathlib
import sys

import munch
import numpy as np
import pandas as pd
import qgrid
from shared import *
from provider_yfinance import *
from dotenv import load_dotenv, find_dotenv
    
def save_config_json(cfg, cfg_path='./config.json'):
    save_config(cfg, cfg_path)


def load_config_json(cfg_path='./config.json'):
    return load_config(cfg_path)


def save_config_yaml(cfg, cfg_path='./config.yaml'):
    save_config(cfg, cfg_path)


def load_config_yaml(cfg_path='./config.yaml'):
    return load_config(cfg_path)


def load_config(cfg_path='./config.json'):
    load_dotenv(find_dotenv(), verbose=True)
    np.random.seed(int(os.environ['RANDOM_SEED']))
    f_cfg = pathlib.Path(cfg_path)
    if f_cfg.is_file():
        with open(f_cfg, 'r', encoding='utf-8') as fd:
            if f_cfg.suffix == '.json':
                cfg = munch.Munch.fromJSON(fd.read())
            else:
                cfg = munch.Munch.fromYAML(fd.read())
            return cfg
    else:
        return None

def save_config(cfg, cfg_path='./config.json'):
    f_cfg = pathlib.Path(cfg_path)
    if f_cfg.suffix == '.json':
        cfg_serialized = cfg.toJSON(indent=4, sort_keys=True)
    else:
        cfg_serialized = cfg.toYAML(allow_unicode=True, default_flow_style=False)
    with open(f_cfg, 'w', encoding='utf-8') as fd:
        fd.write(cfg_serialized)
    print(f"config> saved config to '{f_cfg.resolve()}'")

def save_config(cfg, cfg_path='./config.json'):
    f_cfg = pathlib.Path(cfg_path)
    if f_cfg.suffix == '.json':
        cfg_serialized = cfg.toJSON(indent=4, sort_keys=True)
    else:
        cfg_serialized = cfg.toYAML(allow_unicode=True, default_flow_style=False)
    with open(f_cfg, 'w', encoding='utf-8') as fd:
        fd.write(cfg_serialized)
    print(f"config> saved config to '{f_cfg.resolve()}'")
    
def get_config(selected_index='^GDAXI', overwrite=False, cfg_path=None):
    if cfg_path is None:
        cfg_path = os.environ.get('CONFIG_FILE_PATH', './config.json')

    cfg = load_config(cfg_path)
    if not overwrite and cfg is not None:
        print(f"config> read from config file: '{cfg}'")
        return cfg
    
    # add current directory to sys.path
    p = pathlib.Path('.').resolve()
    current_dir = str(p)
    sys.path.append(current_dir)
    print(f"config> current directory:{current_dir}")
    
    p = pathlib.Path(os.environ.get('CACHE_DIR', './cache/'))
    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    cache_dir = str(p.resolve())    
    p = pathlib.Path(os.environ.get('MODEL_DIR', './model/'))
    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    model_dir = str(p.resolve())    
            
    # feature configurations
    window_trading_days = [int(s.strip()) for s in os.environ['TRAIN_WINDOW_TRADING_DAYS'].strip().split(',')]
    lag_trading_days = [int(s.strip()) for s in os.environ['TRAIN_LAG_TRADING_DAYS'].strip().split(',')]
    max_samples = int(os.environ.get('MODEL_MAX_SAMPLES', '40'))
    max_manifold = int(os.environ.get('TRAIN_MAX_MANIFOLD', '5'))    
    samples_before = int(os.environ.get('TRAIN_PREV_YEAR_SAMPLES_BEFORE', '5'))    
    samples_after = int(os.environ.get('TRAIN_PREV_YEAR_SAMPLES_AFTER', '5'))    
    label_max_high_weight = float(os.environ.get('TRAIN_LABEL_MAX_HIGH_WEIGHT', '3.'))    
    label_max_close_weight = float(os.environ.get('TRAIN_LABEL_MAX_CLOSE_WEIGHT', '1.'))    

    # parse start and end dates
    data_end_dt_str = format_date() if len(os.environ['DATA_END_DT']) == 0 else os.environ['DATA_END_DT']
    end_dt = parse_datetime(data_end_dt_str)
    data_start_dt_str = None if len(os.environ['DATA_START_DT']) == 0 else os.environ['DATA_START_DT']
    if data_start_dt_str is not None:
        start_dt = parse_datetime(data_start_dt_str)
    else:
        weeks = int(os.environ.get('TRAIN_LAST_WEEKS', '30'))        
        start_dt = end_dt - datetime.timedelta(weeks=weeks)
        data_start_dt_str = format_date(start_dt)
    print(f"config> data period: from '{data_start_dt_str}' to '{data_end_dt_str}'")
    
    max_window_trading_days = max(window_trading_days)
    conservative_download_days = math.ceil(7 + (max_window_trading_days / 5. * 7.))
    download_start_dt = start_dt - datetime.timedelta(days=conservative_download_days)
    download_start_dt_str = format_date(download_start_dt)
    download_end_dt_str = data_end_dt_str
    cache_enabled = is_true(os.environ['CACHE_ENABLED'])
    print(f"config> download period: from '{download_start_dt_str}' to '{download_end_dt_str}'")
    
    # read selected features
    if len(os.environ['BENCHMARKS']) > 0:
        benchmarks = sorted([s.upper().strip() for s in os.environ['BENCHMARKS'].strip().split(',')])
    else:
        benchmarks = get_benchmarks()
    if len(os.environ['STOCKS']) > 0:
        stocks = sorted([s.upper().strip() for s in os.environ['STOCKS'].strip().split(',')])
    else:
        stocks = get_stocks(selected_index)
    print(f"config> benchmarks: '{benchmarks}'")
    print(f"config> stocks: '{stocks}'")
        
    # read time horizons
    train_lookback_days = [int(s.strip()) for s in os.environ['TRAIN_LOOKBACK_DAYS'].strip().split(',')]
    train_label_days = [int(s.strip()) for s in os.environ['TRAIN_LABEL_DAYS'].strip().split(',')]
    train_ensemble_weights = [float(s.strip()) for s in os.environ['TRAIN_ENSEMBLE_WEIGHTS'].strip().split(',')]
    train_required_data_days = max(train_lookback_days) + max(train_label_days) + max_samples
    train_start_dt_str = format_date(end_dt - datetime.timedelta(days=train_required_data_days) if len(os.environ['TRAIN_START_DT']) == 0 else os.environ['TRAIN_START_DT'])

    assert data_end_dt_str<=download_end_dt_str, 'data end date after download end date!'
    assert data_start_dt_str>=download_start_dt_str, 'data start date before download start date!'
    assert train_start_dt_str<=data_end_dt_str, 'train start date before data start date!'
    assert train_start_dt_str>=data_start_dt_str, 'train start date before data end date!'
    assert len(train_lookback_days)==len(train_label_days), 'train days config size != test days config size!'
    assert len(train_lookback_days)==len(train_ensemble_weights), 'train days config size != ensemble weights config size!'
    
    train_settings = []
    for i in range(len(train_lookback_days)):
        label_days = train_label_days[i]
        lookback_days = train_lookback_days[i]
        ensemble_weight = train_ensemble_weights[i]        
        train_sample_manifolds = [max(1, min(5, i - lookback_days + min(5, lookback_days)) + 1) for i in range(lookback_days)]
        prev_year_samples_before = samples_before
        prev_year_samples_after = label_days + samples_after
        train_settings.append({
            "lookback_days": lookback_days,
            "label_days": label_days,
            "sample_manifolds": train_sample_manifolds,
            "prev_year_samples_before": prev_year_samples_before,
            "prev_year_samples_after": prev_year_samples_after,
            "ensemble_weight": ensemble_weight
        })
    
    # create config obj
    cfg = munch.munchify({
        'base': {
            'current_dir': current_dir,
            'config_file_path': str(pathlib.Path(cfg_path).resolve()),
            'cache_dir': cache_dir,
            'cache_enabled': cache_enabled,
            'model_dir': model_dir
        },  
        'datasets': {
            'raw': {
                'benchmarks': benchmarks,
                'stocks': stocks
            }
        },
        'model': {
            'max_samples': max_samples,
            'model_dir': f'{model_dir}/{format_build_date(download_end_dt_str)}/',
            'model_base_dir': f'{model_dir}/base/'
        },
        'prepare': {
            'cache_dir': f'{cache_dir}/{format_build_date(download_end_dt_str)}/',
            'download_start_dt': download_start_dt_str,
            'download_end_dt': download_end_dt_str,
            'data_start_dt': data_start_dt_str,
            'data_end_dt': data_end_dt_str
        },
        'train': {            
            'start_dt': train_start_dt_str,
            'end_dt': data_end_dt_str,
            'label_max_high_weight': label_max_high_weight,
            'label_max_close_weight': label_max_close_weight,
            'settings': train_settings,
            'window_trading_days': window_trading_days,
            'lag_trading_days': lag_trading_days,
            'batch_size': 200
        }
    })
    print(f'config> created from .env: {cfg}')
    return cfg

def overwrite_end_dt(cfg, new_end_dt):
    cfg.prepare.download_end_dt = new_end_dt
    cfg.prepare.data_end_dt = new_end_dt
    cfg.train.end_dt = new_end_dt
    cfg.prepare.cache_dir = f'{cfg.base.cache_dir}/{format_build_date(new_end_dt)}/'