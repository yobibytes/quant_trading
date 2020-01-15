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
        if (not ticker_name) or load_model_weights(cfg, mdl, pathlib.Path(cfg.model.model_templates_dir), ticker_name, train_mode=train_mode) is None:
            # try to load current overall weights
            if (not ticker_name) or load_model_weights(cfg, mdl, pth_submodel, train_mode=train_mode) is None:
                # try to load template overall weights
                if load_model_weights(cfg, mdl, pathlib.Path(cfg.model.model_templates_dir), train_mode=train_mode) is None:
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
            raise BaseException(f"model> model overall weights doesn't exists in '{cfg.model.model_templates_dir}'!")
    # try to load current ticker weights
    if load_optimizer_weights(cfg, mdl, pth_submodel, ticker_name, train_mode=train_mode) is None:
        # try to load template + ticker_name
        if (not ticker_name) or load_optimizer_weights(cfg, mdl, pathlib.Path(cfg.model.model_templates_dir), ticker_name, train_mode=train_mode) is None:
            # try to load current overall weights
            if (not ticker_name) or load_optimizer_weights(cfg, mdl, pth_submodel, train_mode=train_mode) is None:
                # try to load template overall weights
                load_optimizer_weights(cfg, mdl, pathlib.Path(cfg.model.model_templates_dir), train_mode=train_mode)

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

def create_model(cfg, submodel_settings, mdl_data, ticker_name='', train_mode=True, learning_rate=0.001):
    # print(f'model> clear backend session')
    K.clear_session()    
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
    optimizer = optimizers.Adam(learning_rate=learning_rate)
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
