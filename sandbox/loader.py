from dotenv import load_dotenv, find_dotenv
import os
import sys
import pathlib
import munch


class Loader():
    def __init__(self, cfg, dfs=None):
        self.cfg = cfg
        self.transform_funcs = shared.find_methods(self, '_transform_', ['funcs'])
        self.filter_funcs = shared.find_methods(self, '_filter_', ['funcs'])
        if dfs is None:
            self.dfs = munch.munchify({})
        else:
            self.dfs = dfs
        print(f'transformers: {sorted(list(self.transform_funcs.keys()))}')
        print(f'filters: {sorted(list(self.filter_funcs.keys()))}')
    
    def _transform_funcs(self, ds_key):
        return self.transform_funcs.get(ds_key, None)

    def _filter_funcs(self, ds_key):
        return self.filter_funcs.get(ds_key, None)

    def _load(self, ds, df, transform_func=None, filter_func=None, debug=True):
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
        cfg = self.cfg
        if 'csv_encoding' in ds:
            encoding = ds.csv_encoding
        else:
            encoding = 'utf-16le'
        if 'csv_sep' in ds:
            sep = ds.csv_sep
        else:
            sep = '\t'
        if 'csv_parse_dates' in ds:
            parse_dates = ds.csv_parse_dates
        else:
            parse_dates = []
        if 'csv_date_fmt' in ds:
            date_fmt = ds.csv_date_fmt
        else:
            date_fmt = None
        return pd.read_csv(
            f'{cfg.paths.raw}/{ds.csv}',
            encoding=encoding,
            sep=sep,
            parse_dates=parse_dates,
            date_parser=shared.get_pandas_dt_parser(date_fmt),
            true_values=['Y'],
            false_values=['N'],
            na_values=['Unavailable', 'not available'],
            low_memory=False
        )

    def _to_source_df(self, ds, df):
        if 'source_columns' in ds:
            return df[ds.source_columns]
        else:
            return self._to_parquet_df(ds, df)

    def _write_parquet(self, ds, df):
        df = self._to_parquet_df(ds, df)
        df.to_parquet(f'{self.cfg.paths.temp}/{ds.parquet}')
        return df

    def _to_parquet_df(self, ds, df):
        if 'parquet_columns' in ds:
            df = df[ds.parquet_columns]
        return df

    def load_raw_datasets(self, write_parquet=False):
        cfg = self.cfg
        dfs = self.dfs
        for key in cfg.datasets.load_order:
            ds = cfg.datasets[key]
            df = None
            if 'csv' in ds:
                print(f"> loading '{key}' csv ...", end='')
                df = self._read_csv(ds)
            elif 'source' in ds:
                s = ds.source
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
            self._write_parquet(cfg.datasets[key], df)
        print(' DONE')
        return dfs

    def load_parquets(self):
        dfs = munch.Munch()
        for key in self.cfg.datasets.load_order:
            print(f"> loading '{key}' parquet ... ", end='')
            dfs[key] = pd.read_parquet(f'{self.cfg.paths.temp}/{self.cfg.datasets[key].parquet}')
        return dfs

