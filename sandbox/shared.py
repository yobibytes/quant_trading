import datetime
import math
import os
import pathlib
import re
import shutil

import numpy as np
import pandas as pd
from numba import jit

SECONDS_IN_MS = 1000
MINUTES_IN_MS = 60 * SECONDS_IN_MS
HOURS_IN_MS = 60 * MINUTES_IN_MS
DAYS_IN_MS = 24 * HOURS_IN_MS
TIMESTAMP_MS_CHECK = 10 ** 10

def parse_datetime(dt_str):
    l = len(dt_str)
    if l == 19:
        if dt_str[10] == ' ':
            return datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        else:
            return datetime.datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S')
    elif l > 19:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%f')
    else:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d')

def get_todays_build_version(version='1.0.0'):
    today_dt = datetime.datetime.combine(datetime.datetime.now(), datetime.time.min)
    build = format_build_datetime(today_dt)
    return version, build, f'{version}_{build}'


def format_datetime(dt=datetime.datetime.now(), fmt='%Y-%m-%d %H:%M:%S'):
    if isinstance(dt, str):
        return dt
    else:
        return dt.strftime(fmt)


def format_build_datetime(dt):
    return format_datetime(dt, '%Y%m%d%H%M%S')


def format_date(dt):
    return format_datetime(dt, '%Y-%m-%d')


def chunks(l, n=200):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def chunks_idx(l, n=200):
    for i in range(0, len(l), n):
        yield i, i + n


def is_true(str):
    if type(str) == bool:
        return str
    else:
        return str.lower() in ['true', '1', 't', 'yes', 'y']


@jit(nopython=True)
def get_daytime(hour_minutes):
    if hour_minutes < 600 or hour_minutes > 2230:
        return 0  # Night
    elif hour_minutes >= 600 and hour_minutes < 830:
        return 1  # Morning
    elif hour_minutes >= 830 and hour_minutes < 1200:
        return 2  # AM
    elif hour_minutes >= 1200 and hour_minutes < 1800:
        return 3  # PM
    else:
        return 4  # Evening


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
