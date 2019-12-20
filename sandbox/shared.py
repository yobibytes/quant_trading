import datetime
import math
import os
import inspect
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
from qgrid import show_grid

STRIP_CHARS = "()[]{}.+*~_-? \t\n\r\x0b\x0c"
SECONDS_IN_MS = 1000
MINUTES_IN_MS = 60 * SECONDS_IN_MS
HOURS_IN_MS = 60 * MINUTES_IN_MS
DAYS_IN_MS = 24 * HOURS_IN_MS
TIMESTAMP_MS_CHECK = 10 ** 10

def left_join(df1, df2, cols1, cols2, keys1=None, keys2=None):
    if keys1 is None:
        keys1 = cols1
    if keys2 is None:
        if len(keys1) == len(cols2):
            keys2 = cols2
        else:
            keys2 = cols1
    return pd.merge(df1[cols1], df2[cols2].drop_duplicates(), how='left', left_on=keys1, right_on=keys2)


def show(df, max_cols=10):
    if len(df.columns) > max_cols:
        print(f'Available columns ({len(df.columns)}) = {df.columns}')
    return show_grid(df[df.columns[:max_cols]])


def apply_filters(cfg, df):
    for c, v in cfg.filters.items():
        if c in df.columns:
            df.drop(df[df[c] != v].index, inplace=True)


def df_to_snake_case(df):
    if df is not None:
        df.rename(to_snake_case, axis='columns', inplace=True)
        return df
    else:
        return df


def format_str(s, lstrip_chars=None):
    if (s is not None) and (s != np.nan) and (s != np.inf) and (s != np.NINF):
        s = str(s).strip(STRIP_CHARS)
        if lstrip_chars is not None:
            s = s.lstrip(lstrip_chars)
        if s.endswith('.0'):
            s = s[:-2]
        if s == '0' or s == '' or s == 'nan' or s == 'None' or s == 'NaN':
            return None
        else:
            return s
    return None


def format_int(s):
    if type(s) == int:
        return s
    else:
        s = format_str(s, '0 ')
        if s is not None:
            return int(s.strip('. '))
        else:
            return None

def format_bool(s):
    if type(s) == bool:
        return s
    else:
        s = format_str(s, '0 ')
        if s is not None:
            return is_true(s)
        else:
            return None
        
def format_float(x):
    x = format_str(x)
    if x is not None:
        return float(x.replace(',', '.'))
    else:
        return np.nan

def df_format_str(ds, df):
    if 'str_columns' in ds:
        for key in ds.str_columns:
            df[key] = df[key].apply(format_str).astype(str)


def df_format_int(ds, df):
    if 'int_columns' in ds:
        for key in ds.int_columns:
            df[key] = df[key].apply(format_int)


def df_format_bool(ds, df):
    if 'bool_columns' in ds:
        for key in ds.bool_columns:
            df[key] = df[key].apply(format_bool).astype(bool)


def df_format_float(ds, df):
    if 'float_columns' in ds:
        for key in ds.float_columns:
            df[key] = df[key].apply(format_float).astype(float)


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
    elif l > 7 :
        return datetime.datetime.strptime(dt_str, '%d-%m-%y')
    elif l > 5:
        return datetime.datetime.strptime(dt_str, '%Y-%m')
    else:
        return datetime.datetime.strptime(dt_str, '%Y')

def get_todays_build_version(version='1.0.0'):
    today_dt = datetime.datetime.combine(datetime.datetime.now(), datetime.time.min)
    build = format_build_datetime(today_dt)
    return version, build, f'{version}_{build}'


def format_datetime(dt=None, fmt=None):
    if dt is None:
        dt = datetime.datetime.now()
    if fmt is None:
        fmt = '%Y-%m-%d %H:%M:%S'
    if isinstance(dt, str):
        return dt
    else:
        return dt.strftime(fmt)


def format_build_datetime(dt=None):
    if type(str) == str:
        return str.replace(':', '')
    else:
        return format_datetime(dt, '%Y%m%d%H%M%S')

def format_build_date(dt=None):
    if type(dt) == str:
        return dt.replace(':', '').replace('-', '')
    else:
        return format_datetime(dt, '%Y%m%d')

def format_date(dt=None):
    return format_datetime(dt, '%Y-%m-%d')


def chunks(l, n=200):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def chunks_idx(l, n=200):
    for i in range(0, len(l), n):
        yield i, i + n

def to_snake_case(s):
    return re.sub('[^a-z0-9_]', '', str(s).strip(STRIP_CHARS).lower().replace(' ', '_'))

def to_snake_cases(df):
    if df is not None:
        df.rename(to_snake_case, axis='columns', inplace=True)
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

    
def is_file(f):
    return f.is_file()


def list_directory(pth, search_pattern='*', filter_func=is_file):
    p = pathlib.Path(pth).glob('*')
    return [f for f in p if filter_func(f)]


def get_pandas_dt_parser(fmt=None):
    if fmt is None:
        fmt = '%d-%m-%y %H:%M:%S'
    parser = lambda x: pd.datetime.strptime(x, fmt) if x and type(x) is str else None
    return parser


def find_methods(obj, prefix, exclusions):
    return dict([
        (n, m) for n, m in [
            (name[len(prefix):], getattr(obj, name)) for name in dir(obj) if name.startswith(prefix)
        ] if inspect.ismethod(m) and n not in exclusions
    ])


class ProviderException(Exception):
    pass


def filter_dates_generator(cfg, cfg_idx, steps=1):
    train_days = cfg.train.train_days[cfg_idx]
    predict_days = cfg.train.test_days[cfg_idx]
    data_end_dt = parse_datetime(cfg.prepare.data_end_dt)    
    train_start_dt = parse_datetime(cfg.train.start_dt)
    while(True):
        train_end_dt = train_start_dt + datetime.timedelta(days=train_days)
        predict_start_dt = train_end_dt + datetime.timedelta(days=1)
        predict_end_dt = predict_start_dt + datetime.timedelta(days=predict_days-1)
        if predict_end_dt > data_end_dt:
            break
        else:
            yield(munch.munchify({
                'train_start_dt': train_start_dt,
                'train_end_dt': train_end_dt,
                'predict_start_dt': predict_start_dt,
                'predict_end_dt': predict_end_dt
            }))
        train_start_dt += datetime.timedelta(days=steps)
        
def is_dataframe(df):
    return df is not None and isinstance(df, pd.DataFrame)

def save_pickle(pth, obj):
    if type(pth) == str:
        f = pathlib.Path(pth)
    else:
        f = pth
    if type(obj) == dict:
        obj_dict = obj
    else:
        obj_dict = munch.unmunchify(obj)
    with f.open('wb') as fp:
        pickle.dump(obj_dict, fp)

def load_pickle(pth):
    if type(pth) == str:
        f = pathlib.Path(pth)
    else:
        f = pth
    if f.is_file():
        with f.open('rb') as fp:
            return munch.munchify(pickle.load(fp))
    return None