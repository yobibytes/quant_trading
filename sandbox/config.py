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
            
    # feature configurations
    window_trading_days = [int(s.strip()) for s in os.environ['TRAIN_WINDOW_TRADING_DAYS'].split(',')]
    lag_trading_days = [int(s.strip()) for s in os.environ['TRAIN_LAG_TRADING_DAYS'].split(',')]
    
    # parse start and end dates
    data_end_dt_str = format_date() if len(os.environ['DATA_END_DT']) == 0 else os.environ['DATA_END_DT']
    end_dt = parse_datetime(data_end_dt_str)
    data_start_dt_str = None if len(os.environ['DATA_START_DT']) == 0 else os.environ['DATA_START_DT']
    if data_start_dt_str is not None:
        start_dt = parse_datetime(data_start_dt_str)
    else:
        weeks = int(os.environ.get('TRAIN_LAST_WEEKS', '12'))        
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
        benchmarks = sorted([s.upper().strip() for s in os.environ['BENCHMARKS'].split(',')])
    else:
        benchmarks = get_benchmarks()
    if len(os.environ['STOCKS']) > 0:
        stocks = sorted([s.upper().strip() for s in os.environ['STOCKS'].split(',')])
    else:
        stocks = get_stocks(selected_index)
    print(f"config> benchmarks: '{benchmarks}'")
    print(f"config> stocks: '{stocks}'")
        
    # read time horizons
    train_start_dt_str = data_start_dt_str if len(os.environ['TRAIN_START_DT']) == 0 else os.environ['TRAIN_START_DT']
    train_train_days = [int(s.strip()) for s in os.environ['TRAIN_TRAIN_DAYS'].split(',')]
    train_test_days = [int(s.strip()) for s in os.environ['TRAIN_TEST_DAYS'].split(',')]    

    assert data_end_dt_str>=download_end_dt_str, 'data end date after download end date!'
    assert data_start_dt_str<=download_start_dt_str, 'data start date before download start date!'
    assert train_start_dt_str>=data_start_dt_str, 'train start date before data start date!'
    assert train_start_dt_str<=data_end_dt_str, 'train start date before data end date!'
    assert len(train_train_days)==len(train_test_days), 'train days config size != test days config size!'
    
    # create config obj
    cfg = munch.munchify({
        'base': {
            'current_dir': current_dir,
            'config_file_path': str(pathlib.Path(cfg_path).resolve()),
            'cache_dir': cache_dir,
            'cache_enabled': cache_enabled
        },  
        'datasets': {
            'raw': {
                'benchmarks': benchmarks,
                'stocks': stocks
            }
        }, 
        'prepare': {
            'download_start_dt': download_start_dt_str,
            'download_end_dt': download_end_dt_str,
            'data_start_dt': data_start_dt_str,
            'data_end_dt': data_end_dt_str
        },
        'train': {            
            'start_dt': train_start_dt_str,
            'train_days': train_train_days,
            'test_days': train_test_days,
            'window_trading_days': window_trading_days,
            'lag_trading_days': lag_trading_days            
        }
    })
    print(f'config> created from .env: {cfg}')
    return cfg
