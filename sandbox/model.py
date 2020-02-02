import collections
import os
import pathlib
import sys

import munch
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K

from shared import *
import config
import provider_yfinance as provider

def load_model_weights(cfg, mdl, pth_model_weights, ticker_name='', train_mode=True):
    if ticker_name:
        pth_model_weights = pth_model_weights.joinpath(ticker_name)
        mkdirs(pth_model_weights)
    f_model_weights = pth_model_weights.joinpath(cfg.model.model_weights_file_name)
    if f_model_weights.is_file():
        try:        
            mdl.load_weights(os.fspath(f_model_weights.resolve()))
            if train_mode:
                print(f"model> loaded model weights from '{f_model_weights}'")
            return mdl
        except ValueError as e:
            print(f"WARN model> failed to load model weights from '{f_model_weights}': ${e}")
    return None

def load_optimizer_weights(cfg, mdl, pth_optimizer_weights, ticker_name='', train_mode=True):
    if ticker_name:
        pth_optimizer_weights = pth_optimizer_weights.joinpath(ticker_name)
        mkdirs(pth_optimizer_weights)
    f_optimizer_weights = pth_optimizer_weights.joinpath(cfg.model.optimizer_weights_file_name)
    if f_optimizer_weights.is_file():
        mdl._make_train_function()
        try:
            with open(f_optimizer_weights.resolve(), 'rb') as f:
                mdl.optimizer.set_weights(pickle.load(f))
                if train_mode:
                    print(f"model> loaded optimizer weights from '{f_optimizer_weights}'")
            return mdl
        except ValueError as e:
            print(f"WARN model> failed to load optimizer weights from '{f_optimizer_weights}': ${e}")
    return None

def load_weights(cfg, submodel_settings, mdl, ticker_name='', train_mode=True):
    pth_submodel = pathlib.Path(f"{cfg.model.base_dir}/{submodel_settings.id}") 
    model_weights_loaded = False
    # try to load current ticker weights
    if load_model_weights(cfg, mdl, pth_submodel, ticker_name, train_mode=train_mode) is None:
        # try to load template + ticker_name
        if (not ticker_name) or load_model_weights(cfg, mdl, pathlib.Path(f"{cfg.model.model_templates_dir}/{submodel_settings.id}"), ticker_name, train_mode=train_mode) is None:
            # try to load current overall weights
            if (not ticker_name) or load_model_weights(cfg, mdl, pth_submodel, train_mode=train_mode) is None:
                # try to load template overall weights
                if load_model_weights(cfg, mdl, pathlib.Path(f"{cfg.model.model_templates_dir}/{submodel_settings.id}"), train_mode=train_mode) is None:
                    model_weights_loaded = 'tpl-overall'
            else:
                model_weights_loaded = 'overall'
        else:            
            model_weights_loaded = 'tpl-ticker'
    else:
        if ticker_name:
            model_weights_loaded = 'ticker'
        else:
            model_weights_loaded = 'overall'
    # print(f'> {model_weights_loaded}')
    if not train_mode:
        if ticker_name and model_weights_loaded != 'ticker':
            raise BaseException(f"model> model ticker weights doesn't exists in '{pth_submodel}'!")
        elif not ticker_name and model_weights_loaded != 'overall':
            raise BaseException(f"model> model overall weights doesn't exists in '{cfg.model.model_templates_dir}/{submodel_settings.id}'!")
    # try to load current ticker weights
    if load_optimizer_weights(cfg, mdl, pth_submodel, ticker_name, train_mode=train_mode) is None:
        # try to load template + ticker_name
        if (not ticker_name) or load_optimizer_weights(cfg, mdl, pathlib.Path(f"{cfg.model.model_templates_dir}/{submodel_settings.id}"), ticker_name, train_mode=train_mode) is None:
            # try to load current overall weights
            if (not ticker_name) or load_optimizer_weights(cfg, mdl, pth_submodel, train_mode=train_mode) is None:
                # try to load template overall weights
                load_optimizer_weights(cfg, mdl, pathlib.Path(f"{cfg.model.model_templates_dir}/{submodel_settings.id}"), train_mode=train_mode)

def save_weights(cfg, submodel_settings, mdl, ticker_name=''):
    print(f"model> trying to save weights ...") 
    pth_submodel = pathlib.Path(f"{cfg.model.base_dir}/{submodel_settings.id}/{ticker_name}")
    f_model_weights = pth_submodel.joinpath(cfg.model.model_weights_file_name)
    f_optimizer_weights = pth_submodel.joinpath(cfg.model.optimizer_weights_file_name)
    mkdirs(pth_submodel)
    mdl.save_weights(os.fspath(f_model_weights))
    print(f"model> saved model weights to '{f_model_weights.resolve()}'")
    with open(f_optimizer_weights.resolve(), 'wb') as f:
        pickle.dump(K.batch_get_value(getattr(mdl.optimizer, 'weights')), f)
        print(f"model> saved optimizer weights to '{f_optimizer_weights.resolve()}'")

def create_model(cfg, submodel_settings, mdl_data=None, ticker_name='', train_mode=True, learning_rate=0.002, input_shape=None):
    # print(f'model> clear backend session')
    K.clear_session()
    if mdl_data is None:
        num_samples = input_shape[0]
        num_features = input_shape[-1]
        input_length = submodel_settings.lookback_days
    else:
        num_samples = mdl_data.shape[0]
        num_features = len(mdl_data.X.head(1).tolist()[0][0][0][0])
        input_length = submodel_settings.lookback_days
    input_dim = num_features
    lstm_dim = cfg.model.lstm_hidden_size
    output_dim = 1
    mdl = Sequential()
    mdl.add(BatchNormalization(input_shape=(input_length, input_dim)))
    mdl.add(Masking())
    mdl.add(LSTM(lstm_dim, dropout=.2, recurrent_dropout=.2, return_sequences=True, activation="softsign"))
    mdl.add(LSTM(lstm_dim, dropout=.2, recurrent_dropout=.2, return_sequences=True, activation="softsign"))
    mdl.add(LSTM(lstm_dim, dropout=.2, recurrent_dropout=.2, activation="softsign"))
    mdl.add(Dense(output_dim))
    
    optimizer = optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    mdl.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])
    if train_mode:
        print(f'model> model created\n:{mdl.summary()}')
    load_weights(cfg, submodel_settings, mdl, ticker_name, train_mode)
    return mdl

def train_model(cfg, submodel_settings, mdl, mdl_data, ticker_name=''):
    num_samples = mdl_data.shape[0]
    num_features = len(mdl_data.X.head(1).tolist()[0][0][0][0])
    input_length = submodel_settings.lookback_days
    input_dim = num_features    
    output_dim = 1
    X = np.hstack(np.asarray(mdl_data.X)).reshape(num_samples, input_length, input_dim)
    y = np.hstack(np.asarray(mdl_data.y)).reshape(num_samples, output_dim)
    pth_submodel = f"{cfg.model.base_dir}/{submodel_settings.id}/{ticker_name}"
    mkdirs(pth_submodel)
    monitor = cfg.model.validation_monitor
    patience = cfg.model.early_stopping_patience
    fit_params = {
        "batch_size": cfg.model.batch_size,
        "epochs": cfg.model.max_epochs,
        "verbose": 1,
        "validation_split": 0.1,
        "shuffle": True,
        "callbacks": [
            EarlyStopping(verbose=True, patience=patience, monitor=monitor),
            ModelCheckpoint(f"{pth_submodel}/best_weights_lstm-{cfg.model.lstm_hidden_size}_epoch-{{epoch:02d}}_val-{{{monitor}:.4f}}.hdf5", monitor=monitor, verbose=1, save_best_only=True)
        ]
    }
    print('model> fitting ... (Hit CTRL-C to stop early)')
    history = None
    try:
        history = mdl.fit(X, y, **fit_params)
    except KeyboardInterrupt:
        print('model> training stopped early!')
        history = mdl.history        
    save_weights(cfg, submodel_settings, mdl, ticker_name)
    return history

def train_full(cfg, start_settings_idx=0):
    monitor = cfg.model.validation_monitor
    patience = cfg.model.early_stopping_patience
    learning_rate = cfg.model.learning_rate
    for submodel_settings in cfg.train.settings[start_settings_idx:]:
        print(f"sm-{submodel_settings.id}> training submodel ...")
        mdl_data = provider.prepare_submodel_data(cfg, submodel_settings)
        mdl = create_model(cfg, submodel_settings, mdl_data, learning_rate=learning_rate)
        history = train_model(cfg, submodel_settings, mdl, mdl_data)
        print(f"sm-{submodel_settings.id}> overall-{monitor} (best epoch): {history.history[monitor][np.max(history.epoch)-patience]}")
        print(f"sm-{submodel_settings.id}> overall-{monitor} (+-5 around best epoch): {np.mean(history.history[monitor][(np.max(history.epoch)-patience-5):(np.max(history.epoch)-patience+5)])}")
        for ticker_name in mdl_data.ticker.unique().tolist():
            ticker_data = mdl_data[mdl_data.ticker==ticker_name]
            mdl = create_model(cfg, submodel_settings, ticker_data, ticker_name)
            history = train_model(cfg, submodel_settings, mdl, ticker_data, ticker_name)
            print(f"sm-{submodel_settings.id}> {ticker_name}-{monitor} (best epoch): {history.history[monitor][np.max(history.epoch)-patience]}")
            print(f"sm-{submodel_settings.id}> {ticker_name}-{monitor} (+-5 around best epoch): {np.mean(history.history[monitor][(np.max(history.epoch)-patience-5):(np.max(history.epoch)-patience+5)])}")        

            
def validate_model(cfg, validate_dt):
    # select model to validate against 
    mdl_cfg = cfg.copy()
    config.overwrite_end_dt(mdl_cfg, validate_dt)    
    eval_result = {}
    verbose=0
    for submodel_settings in cfg.train.settings:
        print(f'============\n {submodel_settings.id}\n ============')
        rs = {}
        mdl_data = provider.prepare_submodel_data(cfg, submodel_settings)
        tickers = mdl_data.ticker.unique().tolist()
        for ticker_name in tickers:        
            ticker_data = mdl_data[(mdl_data.ticker==ticker_name) & (mdl_data.date==mdl_cfg.train.end_dt)]
            base_date = str(ticker_data.date[-1:].tolist()[0].date())
            print(f'eval> {submodel_settings.id} - {ticker_name} - {base_date} ...')
            mdl = create_model(mdl_cfg, submodel_settings, ticker_data, ticker_name, train_mode=False)
            mdl0 = create_model(mdl_cfg, submodel_settings, ticker_data, train_mode=False)
            num_samples = ticker_data.shape[0]    
            num_features = len(ticker_data.X.head(1).tolist()[0][0][0][0])
            input_dim = num_features    
            input_length = submodel_settings.lookback_days
            output_dim = 1
            X = np.hstack(np.asarray(ticker_data.X)).reshape(num_samples, input_length, input_dim)[-1:]
            y = np.hstack(np.asarray(ticker_data.y)).reshape(num_samples, output_dim)[-1:]
            X0 = np.hstack(np.asarray(ticker_data.X)).reshape(num_samples, input_length, input_dim)[-1:]
            y0 = np.hstack(np.asarray(ticker_data.y)).reshape(num_samples, output_dim)[-1:]
            mdl_metrics = dict(zip(mdl.metrics_names, mdl.evaluate(X, y, verbose=verbose)))
            mdl0_metrics = dict(zip(mdl.metrics_names, mdl0.evaluate(X0, y0, verbose=verbose)))
            rs[ticker_name] = {
                'date': [base_date],
                'metrics': [
                    mdl_metrics['loss'], mdl_metrics['mean_absolute_error'], mdl_metrics['mean_squared_error'],
                    mdl0_metrics['loss'], mdl0_metrics['mean_absolute_error'], mdl0_metrics['mean_squared_error'],
                ],
                'y': [round(mdl.predict(X)[0][0]*100)/100, round(y[0][0]*100)/100]
            }
        eval_result[submodel_settings.id] = rs

    csv_output_stocks = []
    rs = ['ticker_name']
    for submodel_settings in cfg.train.settings:
        prefix = submodel_settings.id + '_'
        rs += [
            'date', 'y_predicted', 'y_actual', prefix + 'mdl_loss', prefix + 'mdl_mae', prefix + 'mdl_mse', prefix + 'mdl0_loss', prefix + 'mdl0_mae', prefix + 'mdl0_mse'
        ]
    csv_output_stocks.append(rs)
    for ticker_name in cfg.datasets.raw.stocks:
        rs = [ticker_name]
        for submodel_settings in cfg.train.settings:
            if ticker_name in eval_result[submodel_settings.id]:
                ticker_result = eval_result[submodel_settings.id][ticker_name]
                rs += ticker_result['date']
                rs += ticker_result['y']
                rs += ticker_result['metrics']
            else:
                rs += [None] * 9
        csv_output_stocks.append(rs)  

    with open(os.path.join(cfg.model.base_dir, 'model_eval_pivot.tsv'), 'w', newline='\n', encoding='utf-8') as fp:
        writer = csv.writer(fp, delimiter='\t')
        for rs in csv_output_stocks:
            writer.writerow(rs)

    csv_output = [
        ['ticker_name', 'submodel', 'date', 'y_predicted', 'y_actual', 'mdl_loss', 'mdl_mae', 'mdl_mse', 'mdl0_loss', 'mdl0_mae', 'mdl0_mse']
    ]
    for ticker_name in cfg.datasets.raw.stocks:    
        for submodel_settings in cfg.train.settings:
            rs = [ticker_name, submodel_settings.id]
            if ticker_name in eval_result[submodel_settings.id]:
                ticker_result = eval_result[submodel_settings.id][ticker_name]
                rs += ticker_result['date']
                rs += ticker_result['y']
                rs += ticker_result['metrics']
            else:
                rs += [None] * 9
            csv_output.append(rs)

    with open(os.path.join(cfg.model.base_dir, 'model_eval.tsv'), 'w', newline='\n', encoding='utf-8') as fp:
        writer = csv.writer(fp, delimiter='\t')
        for rs in csv_output:
            writer.writerow(rs)

    return eval_result


def rank_model_by_weighted_score(cfg):
    # rank models by performance
    df_eval = pd.read_csv(os.path.join(cfg.model.base_dir, 'model_eval.tsv'), sep='\t', low_memory=False)
    # ticker model scores
    scores = [10,8,5,3,2,1]
    s_scores = None
    for ticker_name in cfg.datasets.raw.stocks:        
        idx = df_eval[df_eval.ticker_name==ticker_name].sort_values(by='mdl_mae').index    
        scores = scores + [0] * (len(cfg.train.settings) - len(scores))
        if s_scores is None:
            s_scores = pd.Series(scores, index=idx)
        else:
            s_scores = pd.concat([s_scores, pd.Series(scores, index=idx)])
    df_eval['scores'] = s_scores
    # overall model scores
    scores0 = np.array(scores) / 2
    s_scores0 = None
    for ticker_name in cfg.datasets.raw.stocks:        
        idx = df_eval[df_eval.ticker_name==ticker_name].sort_values(by='mdl0_mae').index    
        scores = scores + [0] * (len(cfg.train.settings) - len(scores))
        if s_scores0 is None:
            s_scores0 = pd.Series(scores, index=idx)
        else:
            s_scores0 = pd.concat([s_scores0, pd.Series(scores, index=idx)])
    df_eval['scores0'] = s_scores0
    df_eval['scores_sum'] = df_eval.scores + df_eval.scores0
    df_eval['ensemble_weight'] = df_eval.submodel.apply(lambda x: [s for s in cfg.train.settings if x==s.id][0].ensemble_weight)
    df_eval['scores_weighted'] = df_eval.scores_sum * df_eval.ensemble_weight
    df_rank = df_eval.groupby(['submodel']).agg(sum).sort_values('scores_weighted', ascending=False)[['scores_weighted']]
    return df_rank, df_eval

def predict(cfg, predict_dt=None):
    if predict_dt is None:
        mdl_cfg = cfg.copy()
    else:
        mdl_cfg = cfg.copy()
        config.overwrite_end_dt(mdl_cfg, validate_dt)
    if predict_dt is None:
        end_dt = parse_datetime(cfg.prepare.download_end_dt)
    else:
        end_dt = parse_datetime(predict_dt)
    enc_stocks = provider.encode_stocks(cfg)
    enc_benchmarks = provider.encode_benchmarks(cfg)
    data_stocks_close = provider.load_stocks_close(cfg, end_dt)
    predict_result = {}
    verbose=0
    for submodel_settings in cfg.train.settings:
        print(f'============\n {submodel_settings.id}\n ============')
        rs = {}
        mdl_data = provider.prepare_submodel_data(cfg, submodel_settings)
        tickers = mdl_data.ticker.unique().tolist()
        for ticker_name in tickers:
            ticker_data = enc_stocks[ticker_name]
            ticker_dates = ticker_data.iloc[ticker_data.index <= end_dt].index
            lookback_end_date = ticker_dates[-1]
            lookback_start_date = ticker_dates[-(1 + submodel_settings.lookback_days)]
            back_data = ticker_data.iloc[(ticker_data.index >= lookback_start_date) & (ticker_data.index <= lookback_end_date)]
            # print(f"{submodel_settings.id}> {ticker_name}: prediction lookback from '{lookback_start_date}' to '{lookback_end_date}' for the next {submodel_settings.label_days} days")
            bm_X = provider.generate_relative_benchmark_data(cfg, submodel_settings, enc_benchmarks, lookback_start_date, lookback_end_date)
            # print(f"{submodel_settings.id}> benchmark data shape: {bm_X.shape}")
            X = provider.generate_sample_x(cfg, submodel_settings, ticker_data, bm_X, lookback_start_date, lookback_end_date)
            # print(f"{submodel_settings.id}> X shape: {X.shape}")
            mdl = create_model(cfg, submodel_settings, ticker_name=ticker_name, train_mode=False, input_shape=X.shape)
            prediction = mdl.predict(X)[0][0]
            mdl0 = create_model(cfg, submodel_settings, train_mode=False, input_shape=X.shape)
            prediction0 = mdl0.predict(X)[0][0]
            close_base = data_stocks_close[ticker_name]
            close = close_base * (1.+prediction/100.)
            close0 = close_base * (1.+prediction0/100.)
            print(f"{submodel_settings.id}> {ticker_name}: stock model: {close:.2f}€ ({prediction:.2f}%), index model: {close0:.2f}€ ({prediction0:.2f}%), p0={close_base:.2f}€")
            rs[ticker_name] = [
                submodel_settings.label_days, 
                close_base, 
                close, 
                prediction, 
                close0, 
                prediction0
            ]
        predict_result[submodel_settings.id] = rs