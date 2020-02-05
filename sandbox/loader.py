import collections
import pathlib
import re

import munch
import pandas as pd
import shared


def load_parquets(cfg):
    """
    Iterates over 'load_order' in config and loads all configured 'output_parquet' tables
    :param cfg: Munch config object
    :return: Munch/dict of all parquet output tables
    """
    dfs = munch.Munch()
    for key in cfg.datasets.load_order:
        pth = pathlib.Path(f'{cfg.paths.temp}/{cfg.datasets[key].output_parquet}')
        print(f"> loading '{key}' parquet '{pth}' ... ", end='')
        if pth.is_file():
            dfs[key] = pd.read_parquet(pth)
            print("DONE")
        else:
            print(f"FILE NOT FOUND!")
    return dfs


class Loader():
    """
    Univeral config parser and raw data loader. Can load data from csv, excel and in-memory data sources.

    Common data loading properties are configured in the config.json file. Other more specific ones can be implemented
    using 'transform_<dataset>' or 'filter_<dataset>' interfaces.

    Transformer: transforms the data structure of the dataframe e.g. by adding a new column
    Filter: filter the data by some specific contraints e.g. country='DE'
    """

    def __init__(self, cfg, dfs=None):
        self.cfg = cfg
        self.transform_funcs = shared.find_methods(self, '_transform_', ['funcs'])
        self.filter_funcs = shared.find_methods(self, '_filter_', ['funcs'])
        if dfs is None:
            self.dfs = munch.munchify({})
        else:
            self.dfs = dfs
        # list available transformer functions
        print(f'transformers: {sorted(list(self.transform_funcs.keys()))}')
        # list available filter functions
        print(f'filters: {sorted(list(self.filter_funcs.keys()))}')

    def _transform(self, ds, df):
        # over all datasets
        return df

    def _transform_funcs(self, ds_key):
        return self.transform_funcs.get(ds_key, None)

    def _filter_funcs(self, ds_key):
        return self.filter_funcs.get(ds_key, None)

    def _load(self, ds, df, transform_func=None, filter_func=None, debug=True):
        """
        Main method to load the raw dataset
        :param ds: dataset configuration
        :param df: dataframe of the raw dataset
        :param transform_func: raw dataset specific tranformer
        :param filter_func: raw dataset sepcific filter
        :param debug: debug mode
        :return: extended datafrmae of the raw dataset and contains all columns required for the parquet output
        """
        cfg = self.cfg
        shared.df_to_snake_case(df)
        if 'rename_columns' in ds:
            df.rename(columns=ds.rename_columns, inplace=True)
        if 'datetime_columns' in ds:
            for key, fmt in ds.datetime_columns.items():
                df[key] = pd.to_datetime(df[key], format=fmt)
        if transform_func is not None:
            df = transform_func(ds, df)
        if filter_func is not None:
            df = filter_func(ds, df)
        shared.apply_filters(cfg, df)
        df = self._transform(ds, df)
        if 'primary_keys' in ds:
            n_before = df.shape[0]
            df.drop_duplicates(subset=ds.primary_keys, keep='last', inplace=True)
            n_after = df.shape[0]
            if n_before != n_after:
                if debug:
                    print(f'\n> WARN: duplicated rows found: {n_before - n_after}', end='')
            # df.set_index(ds.primary_keys, inplace=True, verify_integrity=True)
        if 'order_by' in ds:
            df = df.sort_values(by=ds.order_by).reset_index(drop=True)
        shared.df_format_str(ds, df)
        shared.df_format_float(ds, df)
        shared.df_format_int(ds, df)
        shared.df_format_bool(ds, df)
        return df

    def _read_csv(self, ds):
        """
        Read raw dataset from a CSV file
        :param ds: dataset configuration
        :return: dataframe of the raw dataset
        """
        cfg = self.cfg
        if 'input_csv_encoding' in ds:
            encoding = ds.input_csv_encoding
        else:
            encoding = 'utf-16le'
        if 'input_csv_sep' in ds:
            sep = ds.input_csv_sep
        else:
            sep = '\t'
        if 'input_csv_parse_dates' in ds:
            parse_dates = ds.input_csv_parse_dates
        else:
            parse_dates = []
        if 'input_csv_date_fmt' in ds:
            date_fmt = ds.input_csv_date_fmt
        else:
            date_fmt = None
        return pd.read_csv(
            f'{cfg.paths.raw}/{ds.input_csv}',
            encoding=encoding,
            sep=sep,
            parse_dates=parse_dates,
            date_parser=shared.get_pandas_dt_parser(date_fmt),
            true_values=['Y'],
            false_values=['N'],
            na_values=['Unavailable', 'not available'],
            low_memory=False
        )

    def _read_excel(self, ds):
        """
        Read raw dataset from a Excel file
        :param ds: dataset configuration
        :return: dataframe of the raw dataset
        """
        cfg = self.cfg
        if 'input_excel_sheet_name' in ds:
            sheet_name = ds.input_excel_sheet_name
        else:
            sheet_name = 0
        if 'input_excel_header' in ds:
            header = ds.input_excel_header
        else:
            header = 0
        return pd.read_excel(
            f'{cfg.paths.raw}/{ds.input_excel}',
            sheet_name=sheet_name,
            header=header
        )

    def _to_source_df(self, ds, df):
        """
        Read raw dataset from another in-memory dataframe
        :param ds: dataset configuration
        :return: dataframe of the raw dataset
        """
        if 'input_source_columns' in ds:
            # only return a set of pre-defined columns
            return df[ds.input_source_columns]
        else:
            return self._to_parquet_df(ds, df)

    def _write_parquet(self, ds, df):
        """
        Write dataframe to parquet. (Note: columns must have consistent data types.)
        :param ds: dataset configuration
        :param df: extended raw dataset dataframe
        :return: parquet table dataframe
        """
        df = self._to_parquet_df(ds, df)
        df.to_parquet(f'{self.cfg.paths.temp}/{ds.output_parquet}', allow_truncated_timestamps=True)
        return df

    def _to_parquet_df(self, ds, df):
        """
        Prepare parquet table dataframe
        :param ds: dataset configuration
        :param df: extended raw dataset dataframe
        :return: parquet table dataframe.
        """
        if 'output_parquet_columns' in ds:
            df = df[ds.output_parquet_columns]
        return df

    def load_raw_datasets(self, write_parquet=False):
        """
        Iterates over 'load_order' and loads raw datasets one by one.
        :param write_parquet: Flag to write parquet tables
        :return: Munch/dict of all processed raw datasets
        """
        cfg = self.cfg
        dfs = self.dfs
        for key in cfg.datasets.load_order:
            ds = cfg.datasets[key]
            df = None
            if 'input_csv' in ds:
                print(f"> loading '{key}' csv ...", end='')
                df = self._read_csv(ds)
            elif 'input_excel' in ds:
                print(f"> loading '{key}' excel ...", end='')
                df = self._read_excel(ds)
            elif 'input_source' in ds:
                s = ds.input_source
                if type(s) == str:
                    print(f"> creating '{key}' from source ...", end='')
                    df = self._to_source_df(ds, dfs[s]).drop_duplicates()
                elif isinstance(s, collections.Mapping):
                    print(f"> creating '{key}' from join ...", end='')
                    df = shared.left_join(dfs[s.df1], dfs[s.df2], s.cols1, s.cols2, s.keys1, s.keys2).drop_duplicates()
            if df is not None:
                dfs[key] = self._load(ds, df, self._transform_funcs(key), self._filter_funcs(key))
                print(f' {dfs[key].shape}')
        print('> writing parquet files ...', end='')
        for key, df in dfs.items():
            ds = cfg.datasets[key]
            if 'output_parquet' in ds:
                print(f"> writing '{key}' to '{ds.output_parquet}' ...")
                self._write_parquet(ds, df)
        print(' DONE')
        return dfs
