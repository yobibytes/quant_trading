# set env via .env file
import sys
import os
import pandas as pd
import datetime
import time
import munch
import shared
import config
import model
import provider_yfinance as provider

global_start_ts = time.time()

print('init> initialize environment')
config.print_env()

cfg = config.get_config('^GDAXI', True)
# recalc specific date
config.overwrite_end_dt(cfg, '2020-02-26')
config.save_config(cfg)


start_ts = time.time()
print(f'download> start downloading data {cfg.prepare.data_end_dt} ...')
cfg_stocks, data_stocks = provider.load_stocks(cfg, compact=True)
cfg_benchmarks, data_benchmarks = provider.load_benchmarks(cfg, compact=True)
print(f'download> download finished, duration: {time.time() - start_ts:.2f} s')


start_ts = time.time()
print(f'prepare> preparing stock and benchmark data ...')
cfg_stocks, data_stocks = provider.load_stocks(cfg)
cfg_benchmarks, data_benchmarks = provider.load_benchmarks(cfg)

prep_stocks = provider.prepare_stocks(cfg, data_stocks)
prep_benchmarks = provider.prepare_benchmarks(cfg, data_benchmarks)

enc_stocks = provider.encode_stocks(cfg, prep_stocks)
enc_benchmarks = provider.encode_benchmarks(cfg, prep_benchmarks, prep_stocks)

print(f'prepare> preparing submodel data ...')
for submodel_settings in cfg.train.settings:
    print(f"sm-{submodel_settings.id}> preparing submodel data ...")
    model_data = provider.prepare_submodel_data(cfg, submodel_settings, enc_stocks, enc_benchmarks)
    # update num_features setting (informational)
    submodel_settings.num_features = len(model_data.X[0][0][0][0])
config.save_config(cfg)
duration = time.time() - start_ts
print(f'prepare> preparation finished, duration: {duration:.2f} s')

start_ts = time.time()
print(f'train> training started ...')
print('train> list all submodule settings: {[(i, s.id) for i, s in enumerate(cfg.train.settings)]}')
model.train_full(cfg, start_settings_idx=0)
duration = time.time() - start_ts
print(f'train> training finished, duration: {duration:.2f} s')


start_ts = time.time()
print(f'predict> prediction started ...')
predictions = model.predict(cfg)
shared.save_munch(predictions, './predictions.json')
duration = time.time() - start_ts
print(f'predict> prediction finished, duration: {duration:.2f} s')


duration = time.time() - global_start_ts
print(f'run> run finished, duration: {duration:.2f} s')
