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

from importlib import reload
import yfinance_v2 as yf
reload(yf)
yf.pdr_override()

from shared import *

def load_stocks(cfg):
    return _load_tickers(cfg, 'stocks')

def load_benchmarks(cfg, interval='1d'):
    return _load_tickers(cfg, 'benchmarks')

def _download_tickers(cfg, ticker_cfg, interval='1d', retries=10):
    '''
    - period: data period to download (Either Use period parameter or use start and end) Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    - interval: data interval (intraday data cannot extend last 60 days) Valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    - start: If not using period - Download start date string (YYYY-MM-DD) or datetime.
    - end: If not using period - Download end date string (YYYY-MM-DD) or datetime.
    - prepost: Include Pre and Post market data in results? (Default is False)
    - auto_adjust: Adjust all OHLC automatically? (Default is False)
    - actions: Download stock dividends and stock splits events? (Default is True)    
    '''
    retried = 0
    # period = 'max'
    period = '10y'
    print(f"tickers-{ticker_cfg.name}> downloading histories ...")
    while True:
        try:
            df = yf.download(' '.join(ticker_cfg.tickers), period=period, interval=interval, auto_adjust=True, prepost=True, threads=True, proxy=None)
            if df.isnull().values.sum() > 0:
                print(f'WARN: tickers-{ticker_cfg.name}> #missing: {df.isnull().values.sum()}')
                #raise Exception(f"Missing values found! (#missing: {df.isnull().values.sum()})")
            result = {}
            features = sorted(set([k for k, v in df.keys().tolist()]))
            for t in ticker_cfg.tickers:                
                result[t] = {}
                for f in features:
                    if (f,t) in df.columns:
                        result[t][to_category_name(f)] = to_category_names(df[f, t].to_frame(name=f))
            return result
        except Exception as e:
            retried += 1
            if retried > retries:
                raise ProviderException(e)
            else:
                period = {
                    'max': '10y',
                    '10y': '5y',
                    '5y': '2y'
                }.get(period, '2y')
                print(f"WARN: ticker-{ticker_cfg.name}> retrying downloading histories ... (#{retried} of {retries})")
                pass

            
def _data_ticker(cfg, ticker_cfg, t, interval='1d', retries=10):
    retried = 0
    period = 'max'
    print(f"ticker-{t}> loading ticker data ...")
    while True:
        try:
            ticker = yf.Ticker(t)
            df = to_category_names(ticker.history(period=period, interval=interval, prepost=True, actions=True, auto_adjust=True, back_adjust=False, rounding=True, tz=None, proxy=None))
            if df.isnull().values.sum() > 0:
                print(f'WARN: ticker-{t}> #missing: {df.isnull().values.sum()}')
                # raise Exception(f"Missing values found! (#missing: {df.isnull().values.sum()})")
            ticker._history = df
            options = ticker.option_chain()         
            result = {
                'calendar': to_category_names(ticker.get_calendar().T.set_index('Earnings Date')) if ticker.get_calendar() is not None else None,
                'recommendations': to_category_names(ticker.get_recommendations()) if ticker.get_recommendations() is not None else None,
                'info': ticker.get_info(),
                'sustainability': to_category_names(pd.concat([ticker.get_sustainability().T, pd.DataFrame({'Date': [parse_datetime(ticker.get_sustainability().index.name)]}, index=ticker.get_sustainability().T.index)], axis=1).set_index('Date')) if ticker.get_sustainability() is not None else None,
                'earnings': to_category_names(index_to_datetime(ticker.get_earnings(freq='quarterly'), fmt='%Q')) if ticker.get_earnings(freq='quarterly') is not None else None,
                'financials': to_category_names(ticker.get_financials(freq='quarterly').T) if ticker.get_financials(freq='quarterly') is not None else None,
                'balancesheet': to_category_names(ticker.get_balancesheet(freq='quarterly').T) if ticker.get_balancesheet(freq='quarterly') is not None else None,
                'cashflow': to_category_names(ticker.get_cashflow(freq='quarterly').T) if ticker.get_cashflow(freq='quarterly') is not None else None,
                'dividends': to_category_names(ticker.get_dividends().to_frame(name='Dividend')) if ticker.get_dividends() is not None else None,
                'splits': to_category_names(ticker.get_splits().to_frame(name='Stock Split')) if ticker.get_splits() is not None else None,
                'history': df,
                'calls': to_category_names(options.calls.set_index('lastTradeDate') if options.calls is not None else None),
                'puts': to_category_names(options.puts.set_index('lastTradeDate') if options.puts is not None else None),
            }            
            return result
        except Exception as e:
            retried += 1
            if retried > retries:
                raise ProviderException(e)
            else:
                period = {
                    'max': '10y',
                    '10y': '5y',
                    '5y': '2y'
                }.get(period, '2y')
                print(f"WARN: ticker-{t}> retrying loading ticker data ... (#{retried} of {retries})")
                pass

def _data_tickers(cfg, ticker_cfg, interval='1d'):
        result = {}
        for t in ticker_cfg.tickers:
            result[t] = _data_ticker(cfg, ticker_cfg, t, interval)
        return result

def _load_tickers(cfg, key, interval='1d'):
    ticker_cfg = munch.munchify({
        'stocks': {
            'name': 'stocks',
            'tickers':  cfg.datasets.raw.stocks
        },
        'benchmarks': {
            'name': 'benchmarks',
            'tickers':  cfg.datasets.raw.benchmarks
        }
    }.get(key, 'stocks'))
    f_cache = pathlib.Path(f'{cfg.base.cache_dir}/{ticker_cfg.name}_{format_build_date(cfg.prepare.download_start_dt)}_{format_build_date(cfg.prepare.download_end_dt)}_{interval}.pkl')
    if f_cache.is_file():
        with f_cache.open('rb') as fp:
            data = munch.munchify(pickle.load(fp))
    else:
        data_dict = {
            'downloads': _download_tickers(cfg, ticker_cfg, interval),
            'tickers': _data_tickers(cfg, ticker_cfg, interval),
        }
        with f_cache.open('wb') as fp:
            pickle.dump(data_dict, fp)
        data = munch.munchify(data_dict)    
    return ticker_cfg, data

def filter_dates(s, dates, forward_trading_days=0):
    if dates is not None:
        if forward_trading_days>0:
            filter_start_dt = s[s.index <= dates.train_start_dt][-(forward_trading_days+1):-(forward_trading_days)].index[0]
        else:
            filter_start_dt = dates.train_start_dt
        return s[(s.index >= filter_start_dt) & (s.index <= dates.train_end_dt)]
    else:
        return s

def tf_none(s, dates):
    s = filter_dates(s, dates)
    return s

def tf_rel(s, dates):
    s = filter_dates(s, dates)
    return s / s[0]

def _tf_loglag(s, dates, lag_trading_days):
    s = filter_dates(s, dates, lag_trading_days)
    s = np.log(s) - np.log(s.shift(lag_trading_days))
    s = filter_dates(s, dates)
    return s

def tf_loglag(lag_trading_days):
    return lambda s, dates: _tf_loglag(s, dates, lag_trading_days)

def _tf_ma(s, dates, window_trading_days):
    s = filter_dates(s, dates, window_trading_days)
    s = s.rolling(window=window_trading_days).mean()
    s = filter_dates(s, dates)
    return s

def tf_ma(window_trading_days):
    return lambda s, dates: _tf_ma(s, dates, window_trading_days)

def get_ticker_feature(data, ticker, feature, dates=None, ticker_func=tf_none):
    if feature in ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']:
        service = 'tickers'
        feature_type = 'history'
    elif feature in ['close', 'high', 'low', 'open', 'volume']:
        # TODO
        pass
    s = data[service][ticker][feature_type][feature].dropna()
    return ticker_func(s, dates)


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
    