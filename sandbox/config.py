import collections
import os
import pathlib
import sys
import tensorflow as tf
from numba import cuda

import munch
import numpy as np
import pandas as pd
import qgrid
from shared import *
from provider_yfinance import *
from dotenv import load_dotenv, find_dotenv

def print_env():
    # tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(True)

    print("TF version: {}".format(tf.__version__))
    print("Keras version: {}".format(tf.keras.__version__))

    gpu_test = tf.random.uniform([3, 3])
    print(f"Is there a GPU available: {tf.test.is_gpu_available()}")
    print(f"Is the Tensor on GPU #0: {gpu_test.device.endswith('GPU:0')}")
    print(f"Device name: {gpu_test.device}")
    print(f"Eager Execution enabled: {tf.executing_eagerly()}")

    
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
    seed = os.environ['RANDOM_SEED']
    np.random.seed(int(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(int(seed))
    return load_munch(cfg_path)

def save_config(cfg, cfg_path='./config.json'):
    save_munch(cfg, cfg_path)
    
def get_config(selected_index='^GDAXI', overwrite=False, cfg_path=None):
    if cfg_path is None:
        cfg_path = os.environ.get('CONFIG_FILE_PATH', './config.json')

    cfg = load_config(cfg_path)
    if not overwrite and cfg is not None:
        print(f"config> created config from file: '{cfg_path}'")
    else:    
        # add current directory to sys.path
        p = pathlib.Path('.').resolve()
        current_dir = str(p)
        sys.path.append(current_dir)
        print(f"config> current directory:{current_dir}")

        cache_dir = mkdirs(os.environ.get('CACHE_DIR', './cache/'))
        model_dir = mkdirs(os.environ.get('MODEL_DIR', './model/'))
        base_dir = get_file_content(os.path.join(model_dir, '.base'))
        if base_dir is None:
            model_templates_dir = get_latest_subdir(model_dir)
        else:
            model_templates_dir = mkdirs(os.path.join(model_dir, base_dir))

        # feature configurations
        window_trading_days = [int(s.strip()) for s in os.environ['TRAIN_WINDOW_TRADING_DAYS'].strip().split(',')]
        lag_trading_days = [int(s.strip()) for s in os.environ['TRAIN_LAG_TRADING_DAYS'].strip().split(',')]
        max_samples = int(os.environ.get('MODEL_MAX_SAMPLES', '40'))
        max_manifold = int(os.environ.get('TRAIN_MAX_MANIFOLD', '5'))    
        samples_before = int(os.environ.get('TRAIN_PREV_YEAR_SAMPLES_BEFORE', '5'))    
        samples_after = int(os.environ.get('TRAIN_PREV_YEAR_SAMPLES_AFTER', '5'))
        batch_size = int(os.environ.get('TRAIN_BATCH_SIZE', '200'))
        max_epochs = int(os.environ.get('TRAIN_MAX_EPOCHS', '1000'))
        label_max_high_weight = float(os.environ.get('TRAIN_LABEL_MAX_HIGH_WEIGHT', '3.'))    
        label_max_close_weight = float(os.environ.get('TRAIN_LABEL_MAX_CLOSE_WEIGHT', '1.'))   
        early_stopping_patience = int(os.environ.get('TRAIN_EARLY_STOPPING_PATIENCE', '5'))
        lstm_hidden_size = int(os.environ.get('TRAIN_LSTM_HIDDEN_SIZE', '256'))
        learning_rate = float(os.environ.get('TRAIN_LEARNING_RATE', '0.0001'))
        learning_rate_decay = float(os.environ.get('TRAIN_LEARNING_RATE_DECAY', '1.'))
        

        # parse start and end dates
        data_end_dt_str = format_date() if len(os.environ['DATA_END_DT']) == 0 else os.environ['DATA_END_DT']
        end_dt = parse_datetime(data_end_dt_str)
        data_start_dt_str = None if len(os.environ['DATA_START_DT']) == 0 else os.environ['DATA_START_DT']
        if data_start_dt_str is not None:
            start_dt = parse_datetime(data_start_dt_str)
        else:
            weeks = int(os.environ.get('DATA_PREPARE_LAST_WEEKS', '100'))        
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

        submodel_settings = []
        for i in range(len(train_lookback_days)):
            # create sub model settings
            label_days = train_label_days[i]
            lookback_days = train_lookback_days[i]
            ensemble_weight = train_ensemble_weights[i]       
            prev_year_samples_before = samples_before
            prev_year_samples_after = label_days + samples_after
            # reduce train samples of models with longer label days
            submodel_max_samples = max_samples - (label_days * 2)
            # manifold the most recent training samples
            train_sample_manifolds = ([1] * (prev_year_samples_before + prev_year_samples_after)) + [max(1, min(max_manifold, i - submodel_max_samples + min(max_manifold, submodel_max_samples)) + 1)+(i//10) for i in range(submodel_max_samples)]
            submodel_settings.append({
                "id": f"lookback_{lookback_days}-label_{label_days}",
                "lookback_days": lookback_days,
                "label_days": label_days,
                "sample_manifolds": train_sample_manifolds,
                "prev_year_samples_before": prev_year_samples_before,
                "prev_year_samples_after": prev_year_samples_after,
                "float_precision": 100.,
                "ensemble_weight": ensemble_weight,
                "max_samples": submodel_max_samples
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
                'base_dir': f'{model_dir}/{format_build_date(download_end_dt_str)}/',
                'model_templates_dir': model_templates_dir,
                'model_weights_file_name': 'model_weights.hdf5',
                'optimizer_weights_file_name': 'optimizer_weights.pkl',
                'batch_size': batch_size,
                'max_epochs': max_epochs,
                'early_stopping_patience': early_stopping_patience,
                'validation_monitor': 'val_mean_squared_error',
                'lstm_hidden_size': lstm_hidden_size,
                'learning_rate': learning_rate,
                'learning_rate_decay': learning_rate_decay
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
                'settings': submodel_settings,
                'window_trading_days': window_trading_days,
                'lag_trading_days': lag_trading_days                
            }
        })
        save_config(cfg)
        print("config> created from .env")
    
    print(f'''config> config
        - base:
            - config_file_path: {cfg.base.config_file_path}
        - datasets:
            - stocks: {len(cfg.datasets.raw.stocks)}
            - benchmarks: {len(cfg.datasets.raw.benchmarks)}
        - prepare:
            - data_start_dt: {cfg.prepare.data_start_dt}
            - data_end_dt: {cfg.prepare.data_end_dt}
            - cache_dir: {cfg.prepare.cache_dir}
        - train:            
            - window_trading_days: {cfg.train.window_trading_days}
            - lag_trading_days: {cfg.train.lag_trading_days}
            - label_max_high_weight: {cfg.train.label_max_high_weight}
            - label_max_close_weight: {cfg.train.label_max_close_weight}
            - settings: {len(cfg.train.settings)}
        - model:
            - max_samples: {cfg.model.max_samples}
            - batch_size: {cfg.model.batch_size}
            - learning_rate: {cfg.model.learning_rate}
            - learning_rate_decay: {cfg.model.learning_rate_decay}
            - lstm_hidden_size: {cfg.model.lstm_hidden_size}
            - early_stopping_patience: {cfg.model.early_stopping_patience}
            - validation_monitor: {cfg.model.validation_monitor}
            - max_epochs: {cfg.model.max_epochs}
            - base_dir: {cfg.model.base_dir}
            - model_templates_dir: {cfg.model.model_templates_dir}
        ''')
    return cfg

def overwrite_end_dt(cfg, new_end_dt):
    cfg.prepare.download_end_dt = new_end_dt
    cfg.prepare.data_end_dt = new_end_dt
    cfg.train.end_dt = new_end_dt
    cfg.prepare.cache_dir = f'{cfg.base.cache_dir}/{format_build_date(new_end_dt)}/'
    cfg.model.base_dir = f'{cfg.base.model_dir}/{format_build_date(new_end_dt)}/'