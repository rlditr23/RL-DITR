#!/usr/bin/env python3
import os
import time
import hashlib
from pathlib import Path
from functools import lru_cache
import logging

import fire
from tqdm import tqdm
import pandas as pd
import numpy as np

BASE_PATH = Path.cwd()
LOG_PATH = BASE_PATH / 'log'
LOG_PATH.mkdir(parents=True, exist_ok=True)


class Logger(object):
    def __init__(self, log_level=logging.DEBUG):
        self.logname = os.path.join(LOG_PATH, "{}.log".format(time.strftime("%Y-%m-%d")))
        self.logger = logging.getLogger("log")
        self.logger.setLevel(log_level)

        self.formater = logging.Formatter(
            '[%(asctime)s][%(filename)s %(lineno)d][%(levelname)s]: %(message)s')

        self.filelogger = logging.FileHandler(self.logname, mode='a', encoding="UTF-8")
        self.console = logging.StreamHandler()
        self.console.setLevel(log_level)
        self.filelogger.setLevel(log_level)
        self.filelogger.setFormatter(self.formater)
        self.console.setFormatter(self.formater)
        self.logger.addHandler(self.filelogger)
        self.logger.addHandler(self.console)


logger = Logger(log_level=logging.WARNING).logger
logger.info("Start...")

if (__name__ == '__main__') or (__package__ == ''):
    from rl import glu2reward
    from rl import reward2return
else:
    from .rl import glu2reward
    from .rl import reward2return


def hash_func(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def func_args_output_helper(args):
    func, path, output_path, kargs = args
    func(path, output_path, **kargs)


def func_args_stat_helper(args):
    func, path, kargs = args
    return func(path, **kargs)


def parse_long_list(arg):
    """
    parse the long list according to ","
    """
    if isinstance(arg, str):
        if ',' in arg:
            arg = arg.split(',')
        else:
            arg = [arg]
    elif isinstance(arg, tuple):
        arg = list(arg)
    return arg


def datetime_time_norm(sr, times_bin, times_norm, labels=None):
    """
    time part normalization
    """
    times_bin = [pd.Timedelta(t) for t in times_bin]
    td = sr - sr.dt.floor('d')
    sr_norm = pd.cut(td, bins=times_bin, labels=times_norm)
    if labels:
        sr_label = sr_norm.cat.rename_categories(dict(list(zip(times_norm, labels))))
    else:
        sr_label = sr_norm
    sr = sr.dt.date.astype('str') + ' ' + sr_norm.astype('str')
    sr = pd.to_datetime(sr, errors='coerce')
    return sr, sr_label


def ts_complete(df, col_datetime, col_timegroup, freq, n_timegroup, col_name='step'):
    """
    time series completion
    - get a unique time series with two columns: col_datetime and col_timegroup

    col_datetime: str
        date column
    freq: str
        frequency, such as:
        - 'D'
    n_timegroup: int
        number of time aggregates per frequency base, such as:
        - 7
    """
    df = df.copy()
    df[col_timegroup] = df[col_timegroup].astype('int')
    df[col_datetime] = pd.to_datetime(df[col_datetime])

    df['_date'] = df[col_datetime].dt.floor(freq)
    df['_date_order'] = ((df['_date'] - df['_date'].min()) / pd.Timedelta(1, unit=freq)).astype('int')
    df[col_name] = df['_date_order'] * n_timegroup + df[col_timegroup]  # calculate time series order
    df[col_name] = df[col_name].astype('int')
    df = df.drop_duplicates(col_name)  # normal data does not have duplicates
    df = df.set_index(col_name)
    if len(df) > 0:
        df = df.reindex(index=np.arange(df.index.max() + 1))  # complete the time series and complete the sorting
    df = df.drop(columns=['_date', '_date_order'])

    # complete timegroup
    timegroups = np.tile(np.arange(n_timegroup), len(df) // n_timegroup + 1)
    df[col_timegroup] = timegroups[:len(df)]
    return df


def get_datetime_span(dt, deltas=('7D', '7D')):
    """
    get a 14 day time range
    t_50 is the median of the time series
    t_0 is the starting point
    t_100 is the end point
    """
    t_50 = dt.quantile(0.5)
    t_0 = t_50 - pd.Timedelta(deltas[0])
    t_100 = t_50 + pd.Timedelta(deltas[1])
    return t_0, t_100


def read_csv_or_df(path, **kargs):
    """
    read the dataframe or read the dataframe from a csv file
    """
    if isinstance(path, pd.DataFrame):
        df = path
    else:
        df = pd.read_csv(path, **kargs)
    return df


def df2map(df, col_key, col_val):
    return df.drop_duplicates(col_key).set_index(col_key)[col_val]


def move_cols_follow(df, col_anchor, cols_to_follow):
    """
    move cols behind the target column
       c1, c2, c_anchor, ..., c_to_follow (col_to_follow can be located anywhere)
    -> c1, c2, c_anchor, c_to_follow, ...
    """
    if isinstance(cols_to_follow, str):
        cols_to_follow = [cols_to_follow]
    cols = df.columns.drop(cols_to_follow)
    i_anchor = df.columns.get_loc(col_anchor)
    cols = list(cols)
    cols_new = cols[:i_anchor + 1] + cols_to_follow + cols[i_anchor + 1:]
    df_reindex = df.reindex(columns=cols_new)
    return df_reindex


def get_day_freq(t_start, t_end, hour, minute):
    """
    get time index from t_start to t_end
    """
    t_temp = t_start.replace(hour=hour, minute=minute)
    if t_temp < t_start:
        t_temp = t_temp + pd.Timedelta(1, unit='D')
    ts = pd.date_range(start=t_temp, end=t_end, freq='D')
    return ts


def get_drug_time(scheme, start_time, days, hour_points=[8, 12, 16, 20]):
    """
    get indexs of specific four administration times and corresponding drug categories
    """
    t_start = pd.Timestamp(start_time)
    t_end = t_start + pd.Timedelta(days, unit='D')
    ts = []
    for d, h in zip(scheme, hour_points):
        # if d != 'None':
        if d not in ['无','na']:
            ts += [(d, t) for t in get_day_freq(t_start, t_end, hour=h, minute=0)]
    return ts


def make_drug_option(scheme, start_time, days, hour_points=[8, 12, 16, 20]):
    """
    expand insulin information
    """
    points = get_drug_time(scheme, start_time, days, hour_points=hour_points)
    df_option = pd.DataFrame(points, columns=['value', 'datetime'])
    df_option['key'] = 'insulin_group'
    df_option['key_group'] = 'insulin'
    # df_option2 = df_option.copy()
    # df_option2['value'] = 1
    # df_option2['key'] = 'insulin'
    # df_option = pd.concat([df_option, df_option2])
    return df_option


class DiabetesPipeline(object):
    def __init__(self, num_workers=0):
        super(DiabetesPipeline, self).__init__()
        if num_workers is None:
            num_workers = 0
        elif num_workers == 0:
            num_workers = 1
        elif num_workers < 0:
            num_workers = -1
        self._num_workers = num_workers

    def _std_df_output(self, df, output_path, with_suffix=None):
        if output_path is None:
            return df
        elif output_path == 'print':
            print(df)
        elif output_path == 'skip':
            return
        else:
            output_path = Path(output_path)
            if with_suffix:
                output_path = output_path.with_suffix(with_suffix)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

    @lru_cache(10)
    def _get_drug_meta(self, path):
        """
        get various drug information
        drug_meta: all drug information
        drug_name2group: list of categories to which the drug belongs
        drug_name2min: reasonable min value of corresponding drug
        drug_name2max: reasonable max value of corresponding drug
        """

        # drug
        df_drug_meta = pd.read_excel(path)
        # df_drug_meta = df_drug_meta[df_drug_meta['drug'].notna()]
        df_drug_meta = df_drug_meta[df_drug_meta['药物类'].notna()]
        # df_drug_meta = df_drug_meta.rename(columns={'drug': 'drug_type'})
        df_drug_meta = df_drug_meta.rename(columns={'药物类': 'drug_type'})
        # df_drug_meta['reasonable min'] = df_drug_meta['reasonable min'].fillna(0)
        df_drug_meta['合理min'] = df_drug_meta['合理min'].fillna(0)
        # drug_name2group = df2map(df_drug_meta, 'medication (common Name)', 'drug_type')
        drug_name2group = df2map(df_drug_meta, '用药（通用名称）', 'drug_type')
        # drug_name2min = df2map(df_drug_meta, 'medication (common Name)', 'reasonable min')
        drug_name2min = df2map(df_drug_meta, '用药（通用名称）', '合理min')
        # drug_name2max = df2map(df_drug_meta, 'medication (common Name)', 'reasonable max')
        drug_name2max = df2map(df_drug_meta, '用药（通用名称）', '合理max')

        df_drug_meta.head(2)
        df_drug_meta.sort_values('size', ascending=False)
        return df_drug_meta, drug_name2group, drug_name2min, drug_name2max

    def pipeline1_drug(self, input_path, output_path, drug_meta_path, verbose=0):
        '''
        process the medication information in the input_path file
        '''

        df_drug_meta, drug_name2group, drug_name2min, drug_name2max = self._get_drug_meta(drug_meta_path)

        df_drug = read_csv_or_df(input_path)
        df_drug_output = df_drug

        # drug category mapping
        df_drug_output['drug_group'] = df_drug_output['drug_name'].map(drug_name2group)
        if verbose:
            pass

        # If the category value lacks, but the name contains insulin, it is classified as insulin.
        df_drug_output['drug_group'] = df_drug_output['drug_group'].mask(
            # df_drug_output['drug_group'].isna() & df_drug_output['drug_name'].str.contains('insulin'), 'insulin')
            df_drug_output['drug_group'].isna() & df_drug_output['drug_name'].str.contains('胰岛素'), '胰岛素')

        # remove data that is not in the drug list
        df_drug_output_t = df_drug_output.dropna(subset=['drug_group'])
        df_drug_output = df_drug_output_t

        # remove doctor's advice with unreasonable medication values
        min_ok = df_drug_output['value'] >= df_drug_output['drug_name'].map(drug_name2min)
        max_ok = df_drug_output['value'] <= df_drug_output['drug_name'].map(drug_name2max)
        # df_drug_output_t = df_drug_output[(min_ok & max_ok) | df_drug_output['drug_group'].isin(['insulin'])]
        df_drug_output_t = df_drug_output[(min_ok & max_ok) | df_drug_output['drug_group'].isin(['胰岛素'])]
        df_drug_output = df_drug_output_t

        logger.info("medication information processing completed, in total %s items" % df_drug_output.shape[0])

        return self._std_df_output(df_drug_output, output_path)

    @lru_cache(10)
    def _get_insulin_meta(self, path):
        '''
        get various information about insulin
        df_insulin_meta: all insulin information
        insulin2group: category table of insulin
        insulin_list: insulin list
        '''

        df_insulin_meta = pd.read_excel(path)
        # df_insulin_meta['category brief'] = df_insulin_meta['category'].str.replace('like', '')
        df_insulin_meta['类别简'] = df_insulin_meta['类别'].str.replace('类似物',
                                                                   '')  # In terms of category, xxx analogues and xxx are considered to be the same category of drugs
        # insulin2group = df2map(df_insulin_meta, 'drug name', 'category brief')
        insulin2group = df2map(df_insulin_meta, '药名', '类别简')
        # insulin_list = ['None'] + list(sorted(insulin2group.dropna().unique()))
        insulin_list = ['无'] + list(sorted(insulin2group.dropna().unique()))
        return df_insulin_meta, insulin2group, insulin_list

    def pipeline1_insulin(self, input_path, output_path, drug_meta_path, insulin_meta_path):
        '''
        process the insulin information in the input_path file
        '''

        df_drug_meta, drug_name2group, drug_name2min, drug_name2max = self._get_drug_meta(drug_meta_path)
        df_insulin_meta, insulin2group, insulin_list = self._get_insulin_meta(insulin_meta_path)

        df_insulin = read_csv_or_df(input_path)
        df_insulin_output = df_insulin

        # remove doctor's advice with unreasonable values
        df_insulin_output = df_insulin_output[df_insulin_output['value'].between(1, 50)]
        df_insulin_output = df_insulin_output[df_insulin_output['drug_group'].isin(['胰岛素', 'insulin'])]

        # supplementary drug category Insulin
        if df_insulin_output['key'].isin(['insulin_group']).sum() == 0:  # avoid duplicate additions
            df_insulin_output2 = df_insulin_output.copy()
            df_insulin_output2['key'] = 'insulin_group'
            df_insulin_output2['key_type'] = 'cat{}({})'.format(len(insulin_list), ','.join(insulin_list))
            df_insulin_output2['key_group'] = 'insulin'
            df_insulin_output2['value'] = df_insulin_output2['drug_name'].map(insulin2group)
            if df_insulin_output2['value'].isna().sum() > 0:
                df_na = df_insulin_output2[df_insulin_output2['value'].isna()]
                # df_insulin_output2['value'] = df_insulin_output2['value'].fillna('premixed insulin')
                df_insulin_output2['value'] = df_insulin_output2['value'].fillna('预混胰岛素')
                names = list(df_na['drug_name'].unique())
                if isinstance(input_path, str) or isinstance(input_path, Path):
                    logger.warning(f"insulin mapping missing, unknown short/premixed/long {input_path} {names}")
                else:
                    logger.warning(f"insulin mapping missing, unknown short/premixed/long {df_na.head(2)} {names}")

            df_insulin_output = pd.concat([df_insulin_output, df_insulin_output2])

        # remove missing values
        df_insulin_output = df_insulin_output.dropna(subset=['value'])

        logger.info("insulin information processing completed, %s items in total" % df_insulin_output.shape[0])

        return self._std_df_output(df_insulin_output, output_path)

    @lru_cache(10)
    def _get_test_meta(self, path):
        '''
        get various inspection information
        df_test_meta: all inspection information
        key2name: Mapping table of inspection and corresponding name
        name2lonic: Mapping table of name and corresponding loinc information
        '''

        df_test_meta = pd.read_excel(path)
        cols_test_meta = 'system feature_name loinc component short_name'.split()
        cols_local = df_test_meta.columns.drop(cols_test_meta)

        key2name = df_test_meta.melt(id_vars='feature_name', value_vars=cols_local, var_name='key_map',
                                     value_name='local_name')
        key2name = key2name.drop_duplicates(subset=['key_map', 'local_name'])
        key2name = key2name.dropna()
        key2name = key2name.set_index(['key_map', 'local_name'])['feature_name']

        name2loinc = df2map(df_test_meta, 'feature_name', 'loinc')
        return df_test_meta, key2name, name2loinc

    def pipeline1_test(self, input_path, output_path, test_meta_path):
        '''
        process the inspection information in document input_path
        '''

        df_test_meta, key2name, name2loinc = self._get_test_meta(test_meta_path)

        df_test = read_csv_or_df(input_path)
        df_test_output = df_test

        # inspection index naming standardization
        temp = pd.Series(zip(df_test_output['key_map'], df_test_output['key']))
        temp = temp.map(key2name)
        temp = temp.fillna(df_test_output['key'])
        df_test_output['key'] = temp
        df_test_output['comment'] = df_test_output['key'].map(name2loinc)

        logger.info("inspection information processing completed, %s items in total" % df_test_output.shape[0])

        return self._std_df_output(df_test_output, output_path)

    def workflow1_merge(self, input_dir, output_path,
                        drug_meta_path, insulin_meta_path, test_meta_path,
                        pipeline=['base', 'sym', 'diag', 'drug', 'insulin', 'glu', 'test'],
                        mapping=True,
                        ):
        '''
        integrate multiple data information
        - Input directory, output file csv
        '''

        logger.info("start merging patient information")
        if isinstance(input_dir, str) or isinstance(input_dir, Path):
            input_dir = Path(input_dir)

            path_base = input_dir / 'base.sparse.csv'
            path_sym = input_dir / 'sym.sparse.csv'
            path_diag = input_dir / 'diag.sparse.csv'
            path_drug = input_dir / 'drug.sparse.csv'
            path_insulin = input_dir / 'insulin.sparse.csv'
            path_glu = input_dir / 'glu.sparse.csv'
            path_test = input_dir / 'test.sparse.csv'
        else:
            path_base, path_glu, path_diag, path_test, path_drug, path_insulin, path_sym = input_dir

        dfs = []
        if isinstance(input_dir, str) or isinstance(input_dir, Path):
            input_dir = Path(input_dir)
            if ('base' in pipeline) and path_base.exists():
                logger.info("process base")
                df_base = read_csv_or_df(path_base)
                dfs += [df_base]
            if ('sym' in pipeline) and path_sym.exists():
                logger.info("process sym")
                df_sym = read_csv_or_df(path_sym)
                dfs += [df_sym]
            if ('glu' in pipeline) and path_glu.exists():
                logger.info("process glu")
                df_glu = read_csv_or_df(path_glu)
                dfs += [df_glu]
            if ('diag' in pipeline) and path_diag.exists():
                df_diag = read_csv_or_df(path_diag)
                dfs += [df_diag]
            if ('drug' in pipeline) and path_drug.exists():
                logger.info("process drug")
                if mapping:
                    df_drug = self.pipeline1_drug(path_drug, None, drug_meta_path=drug_meta_path)
                else:
                    df_drug = read_csv_or_df(path_drug)
                dfs += [df_drug]
            if ('insulin' in pipeline) and path_insulin.exists():
                logger.info("process insulin")
                if mapping:
                    df_insulin = self.pipeline1_insulin(path_insulin, None, drug_meta_path=drug_meta_path,
                                                        insulin_meta_path=insulin_meta_path)
                else:
                    df_insulin = read_csv_or_df(path_insulin)
                dfs += [df_insulin]
            if ('test' in pipeline) and path_test.exists():
                logger.info("process test")
                if mapping:
                    df_test = self.pipeline1_test(path_test, None, test_meta_path=test_meta_path)
                else:
                    df_test = read_csv_or_df(path_test)
                dfs += [df_test]
        else:
            if ('base' in pipeline):
                df_base = read_csv_or_df(path_base)
                dfs += [df_base]
            if ('sym' in pipeline):
                df_sym = read_csv_or_df(path_sym)
                dfs += [df_sym]
            if ('glu' in pipeline):
                df_glu = read_csv_or_df(path_glu)
                dfs += [df_glu]
            if ('diag' in pipeline):
                df_diag = read_csv_or_df(path_diag)
                dfs += [df_diag]
            if ('drug' in pipeline):
                df_drug = self.pipeline1_drug(path_drug, None, drug_meta_path=drug_meta_path)
                dfs += [df_drug]
            if ('insulin' in pipeline):
                df_insulin = self.pipeline1_insulin(path_insulin, None, drug_meta_path=drug_meta_path,
                                                    insulin_meta_path=insulin_meta_path)
                dfs += [df_insulin]
            if ('test' in pipeline):
                df_test = self.pipeline1_test(path_test, None, test_meta_path=test_meta_path)
                dfs += [df_test]

        if len(dfs) == 0:
            logger.warning(f"{input_dir} no records were found for this patient")
            df = pd.DataFrame()
        else:
            df = pd.concat(dfs)

        if output_path is None:
            return df
        elif output_path == 'print':
            print(df)
        else:
            output_path = Path(output_path).with_suffix('.csv')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

    def pipeline2ext_add_option(self, input_path, output_path, scheme, start_time, days):
        '''
        add action option
        '''
        df_sample = read_csv_or_df(input_path)

        df_option = make_drug_option(scheme, start_time, days)
        df_sample = pd.concat([df_sample, df_option])

        return self._std_df_output(df_sample, output_path)

    def pipeline2_tnorm(self, input_path, output_path, tnorm_mode):
        '''
        time point normalization
        '''

        df_sample = read_csv_or_df(input_path)
        df_sample_tnorm = df_sample

        if tnorm_mode == 'arm':
            times_bin = '04:30:00 10:00:00 15:00:00 18:50:00 23:59:59'.split()
            times_norm = '06:00:00 10:30:00 16:30:00 21:00:00'.split()
            labels = '0 2 4 6'.split()

        elif tnorm_mode == 'fulltime':
            times_bin = '04:30:00 08:11:00 09:30:00 12:11:00 15:00:00 18:00:00 20:00:00 23:59:59'.split()
            times_norm = '06:00:00 08:30:00 10:30:00 13:00:00 16:30:00 19:00:00 21:00:00'.split()
            labels = '0 1 2 3 4 5 6'.split()
        df_sample_tnorm['datetime'] = pd.to_datetime(df_sample_tnorm['datetime'])
        sr_datetime, sr_datetime_label = datetime_time_norm(df_sample_tnorm['datetime'], times_bin, times_norm, labels)
        df_sample_tnorm['datetime_norm'] = sr_datetime
        df_sample_tnorm['timegroup'] = sr_datetime_label
        df_sample_tnorm = move_cols_follow(df_sample_tnorm, 'datetime', ['datetime_norm', 'timegroup'])
        logger.info("time point normalization completed")
        return self._std_df_output(df_sample_tnorm, output_path)

    @lru_cache(10)
    def _get_feat_meta(self, path):
        '''
        obtain all feature information 
        and segment features based on continuous and discrete attributes
        '''
        if isinstance(path, pd.DataFrame):
            df_feat_meta = path
        else:
            df_feat_meta = pd.read_csv(path)
        df_meta_to_cont = df_feat_meta[df_feat_meta['key_type'] == 'cont']
        df_meta_to_cat = df_feat_meta[df_feat_meta['key_type'].str.contains('^cat')]
        return df_feat_meta, df_meta_to_cont, df_meta_to_cat

    def pipeline2_ftcrop(self, input_path, output_path, feat_meta_path,
                         keygroup_specific=[], dt_deltas=('14D', '14D'),
                         col_datetime='datetime_norm',
                         ):
        '''
        feature and time tailoring
        '''

        df_feat_meta, df_meta_to_cont, df_meta_to_cat = self._get_feat_meta(feat_meta_path)

        df_sample = read_csv_or_df(input_path)

        # feature tailoring
        df_sample_crop = df_sample
        if not ('feat_name' in df_sample_crop):
            df_sample_crop['feat_name'] = df_sample_crop['key_group'] + '|' + df_sample_crop['key']
        feat_names = df_sample_crop['feat_name'].tolist() + ['insulin|insulin', 'insulin|insulin_group']
        df_sample_crop = df_sample_crop[df_sample_crop['feat_name'].isin(feat_names)]

        # time tailoring
        if len(keygroup_specific) > 0:
            dt = df_sample_crop[col_datetime][df_sample_crop['key_group'].isin(keygroup_specific)]
        else:
            dt = df_sample_crop[col_datetime]

        dt_begin, dt_end = get_datetime_span(dt, deltas=dt_deltas)
        df_sample_crop = df_sample_crop[df_sample_crop['datetime_norm'].between(dt_begin, dt_end)]

        logger.info("feature filtering completed")

        return self._std_df_output(df_sample_crop, output_path)

    def pipeline2_ts_flatten(self, input_path, output_path,
                             max_steps=None,
                             freq='D', n_timegroup=7, max_duration='60D',
                             col_datetime='datetime_norm', col_timegroup='timegroup',
                             # col_key='key',col_value='value',col_key_type='key_type',
                             col_key_group='key_group', col_key='key', col_value='value',
                             ):
        '''
        Need to be completed before time flattening:
        - time point normalization
        - feature filtering
        - time span truncation

        '''

        df_sample = read_csv_or_df(input_path)
        df_ts = df_sample
        df_ts[col_datetime] = pd.to_datetime(df_ts[col_datetime])

        # rename features(name + type)
        df_ts[col_key] = df_ts[col_key_group] + '|' + df_ts[col_key]

        # flatten features
        df_ts = df_ts.pivot_table(index=[col_datetime, col_timegroup], columns=col_key, values=col_value,
                                  aggfunc='first')
        df_ts = df_ts.reset_index()
        df_ts.columns.name = None

        # sequential normalization
        tmin, tmax = df_ts[col_datetime].min(), df_ts[col_datetime].max()
        max_duration = pd.Timedelta(max_duration)
        if (tmax - tmin).days > max_duration.days:
            raise Exception(f'Data duration {tmax - tmin} exceeds the limitation of {max_duration}')
        if len(df_ts) > 0:
            df_ts = ts_complete(df_ts, col_datetime, col_timegroup, freq, n_timegroup)
        df_ts = df_ts.reset_index(drop=False)

        # truncation
        if max_steps is not None:
            n_interval = n_timegroup
            if n_interval is None:
                n_interval = 1
            weight = df_ts.notna().sum(axis=1) - 3
            weight = weight / max(weight.sum(), 1)
            pos = (weight * np.arange(len(weight))).sum()
            index_s = max(0, int(pos - max_steps // 2))
            index_s = index_s // n_interval * n_interval
            index_e = min(index_s + max_steps, len(df_ts))
            df_ts = df_ts.iloc[index_s:index_e]

        logger.info("time flattening completed")
        return self._std_df_output(df_ts, output_path)

    def workflow2_ts(self, input_path, output_path, feat_meta_path, keygroup_specific=['insulin', 'glu'],
                     dt_deltas=('14D', '14D')):
        '''
        normalize features and time into sequential features
        '''

        logger.info("start acquiring sequential feature")

        df_sample = read_csv_or_df(input_path)

        key_groups = ['drug', 'insulin']
        df_sample_base = df_sample[~df_sample['key_group'].isin(key_groups)].copy()
        df_sample_drugs = df_sample[df_sample['key_group'].isin(key_groups)].copy()
        df_sample_base = self.pipeline2_tnorm(df_sample_base, None, tnorm_mode='fulltime')
        df_sample_drugs = self.pipeline2_tnorm(df_sample_drugs, None, tnorm_mode='arm')
        df_sample = pd.concat([df_sample_base, df_sample_drugs])
        df_sample = self.pipeline2_ftcrop(df_sample, None, feat_meta_path=feat_meta_path,
                                          keygroup_specific=keygroup_specific, dt_deltas=dt_deltas,
                                          col_datetime='datetime_norm', )
        df_sample = self.pipeline2_ts_flatten(df_sample, None, col_datetime='datetime_norm')
        for col in df_sample.columns:
            if df_sample[col].dtype.name in ['int64', 'datetime64[ns]']:
                continue
            df_sample[col] = pd.to_numeric(df_sample[col], errors='ignore')

        return self._std_df_output(df_sample, output_path)

    def pipeline31_fillna(self, input_path, output_path, feat_meta_path, return_mask=False):
        '''
        zero padding according to the continuous and discrete nature of the feature
        '''

        logger.info("fill feature with zero")

        df_feat_meta, df_meta_to_cont, df_meta_to_cat = self._get_feat_meta(feat_meta_path)

        df_sample = read_csv_or_df(input_path)

        df_case = df_sample
        df_case = df_case.reindex(columns=df_feat_meta['feat_name'])
        df_mask = df_case.notna()

        for _, sr_col in df_feat_meta.iterrows():
            col = sr_col['feat_name']
            interp = sr_col['interp']
            feat_type = sr_col['key_type']
            val = df_case[col]
            if interp == 'ffill':
                val = val.interpolate(method='ffill')
                val = val.interpolate(method='bfill')
            val = val.fillna(0)

            df_case[col] = val
        if return_mask:
            return df_case, df_mask
        else:
            return self._std_df_output(df_case, output_path)

    def pipeline31_delay(self, input_path, output_path, feat_meta_path):
        # delayed observation
        df_sample = read_csv_or_df(input_path)
        df_case = df_sample

        df_feat_meta, df_meta_to_cont, df_meta_to_cat = self._get_feat_meta(feat_meta_path)

        for _, sr_col in df_feat_meta[['feat_name', 'delay']].iterrows():
            col = sr_col['feat_name']
            delay = sr_col['delay']

            val = df_case[col]
            if (delay > 0) and (len(val) > 0):
                v = np.concatenate([np.zeros([delay], dtype=val.dtype), val.values[:-delay]])
                val = pd.Series(v, index=val.index)

            df_case[col] = val
        return df_case

    def pipeline31_onehot(self, input_path, output_path, feat_meta_path, feature_reindex=True):
        '''
        process features, expand discrete features with one hot
        '''

        logger.info("expand discrete features")

        df_sample = read_csv_or_df(input_path)
        df_feat_meta, df_meta_to_cont, df_meta_to_cat = self._get_feat_meta(feat_meta_path)

        df_case = df_sample
        feats = []
        if feature_reindex:
            feats += [df_case.reindex(columns=df_meta_to_cont['feat_name'])]
        else:
            cols_cont = df_case.columns[df_case.columns.isin(df_meta_to_cont['feat_name'])]
            feats += [df_case.reindex(columns=cols_cont)]

        if feature_reindex:
            df_meta_to_cat_t = df_meta_to_cat
        else:
            df_meta_to_cat_t = df_meta_to_cat[df_meta_to_cat['feat_name'].isin(df_case.columns)]

        for _, sr_col in df_meta_to_cat_t.iterrows():
            col = sr_col['feat_name']
            vals = df_case[col]
            if not pd.isna(sr_col['cat']):
                cats = sr_col['cat'].split(',')
                cat2id = {c: i for i, c in enumerate(cats)}
                vals = vals.map(cat2id)
            vals = pd.Categorical(vals, categories=np.arange(sr_col['n_dim']))
            vals = pd.get_dummies(vals)
            vals = vals.add_prefix(col + '_')
            feats += [vals]
        df_case = pd.concat(feats, axis=1)

        return self._std_df_output(df_case, output_path)

    def workflow31_preprocess(self, input_path, output_path, feat_meta_path,
                              add_mask=False,
                              add_timegroup=False, col_timegroup='timegroup', n_timegroup=7,
                              ):
        '''
        sequential feature preprocessing(filling, delay, discrete features, one hot)

        - fill missing value
        - feature delay
        - categorical feature expansion
        - the time dimension remains unchanged, the feature dimension changes
        '''

        logger.info("start sequential feature preprocessing")
        df_case = read_csv_or_df(input_path)
        df_feat, df_mask = self.pipeline31_fillna(df_case, None, feat_meta_path, return_mask=True)
        df_feat = self.pipeline31_onehot(df_feat, None, feat_meta_path, feature_reindex=True)

        # add mask feature
        if add_mask:
            df_feat = pd.concat([df_feat, df_mask.add_suffix('_notna').astype('int')], axis=1)

        # add timegroup tag feature
        if add_timegroup:
            df_time = pd.Categorical(df_case[col_timegroup], categories=np.arange(n_timegroup))
            df_time = pd.get_dummies(df_time)
            df_feat = pd.concat([df_feat, df_time.add_prefix('timegroup_').astype('int')], axis=1)

        return self._std_df_output(df_feat, output_path)

    def pipeline32_reward(self, input_path, output_path, col_glu='glu|glu', reward_dtype='risk', gamma=0.9):
        '''
        add reward return
        '''
        df_sample = read_csv_or_df(input_path)
        df_case_labeled = df_sample
        df_case_labeled['reward'] = glu2reward(df_case_labeled[col_glu], dtype=reward_dtype)
        df_case_labeled['return'] = reward2return(df_case_labeled['reward'], gamma=gamma)
        df_case_labeled['_col_glu'] = df_case_labeled[col_glu]

        return self._std_df_output(df_case_labeled, output_path)

    def pipeline32_action(self, input_path, output_path,
                          col_insulin_dose='insulin|insulin', insulin_max=49,
                          col_insulin_cat='insulin|insulin_group', option_max=4,
                          ):
        '''
        add action option
        match five types of insulin to 1-3
        '''
        option_map = {
            '短效胰岛素': 1,
            '速效胰岛素': 1,
            '中效胰岛素': 2,
            '预混胰岛素': 2,
            '长效胰岛素': 3,
            'Short-acting insulin': 1,
            'Rapid-acting insulin': 1,
            'Medium-acting insulin': 2,
            'Premixed insulin': 2,
            'Long-acting insulin': 3,
            'short': 1,
            'rapid': 1,
            'medium': 2,
            'premixed': 2,
            'long': 3,
        }

        df_sample = read_csv_or_df(input_path)
        df_case_labeled = df_sample
        if (col_insulin_dose not in df_case_labeled) and (col_insulin_cat not in df_case_labeled):
            logger.warning(f"no insulin data input")
            pass
        if col_insulin_dose in df_case_labeled:
            # interpolate dose with zero and normalize it
            df_case_labeled['action'] = df_case_labeled[col_insulin_dose].fillna(0).astype('int')
            df_case_labeled['action'] = df_case_labeled['action'].clip(0, insulin_max)
        else:
            df_case_labeled['action'] = 0

        if col_insulin_cat in df_case_labeled:
            # map, supplement, and normalize options
            df_case_labeled['option'] = df_case_labeled[col_insulin_cat].map(option_map).fillna(0).astype('int')
            df_case_labeled['option'] = df_case_labeled['option'].clip(0, option_max)
        else:
            df_case_labeled['option'] = 0

        return self._std_df_output(df_case_labeled, output_path)

    def pipeline32_glu_label(self, input_path, output_path,
                             col_glu='glu|glu',
                             glu_min=3.9, glu_max=10,
                             n_lookaheads=[7, 14, 21], labels=['glu<3', 'glu>11', '3>glu>11']
                             ):

        df_sample = read_csv_or_df(input_path)
        df_case_labeled = df_sample

        df_case_labeled['glu<3'] = (df_case_labeled[col_glu] < glu_min).astype('int')
        df_case_labeled['glu<3'] = df_case_labeled['glu<3'].mask(df_case_labeled[col_glu].isna(), np.nan)
        df_case_labeled['3>glu>11'] = (~df_case_labeled[col_glu].between(glu_min, glu_max)).astype('int')
        df_case_labeled['3>glu>11'] = df_case_labeled['3>glu>11'].mask(df_case_labeled[col_glu].isna(), np.nan)
        df_case_labeled['glu>11'] = (df_case_labeled[col_glu] > glu_max).astype('int')
        df_case_labeled['glu>11'] = df_case_labeled['glu>11'].mask(df_case_labeled[col_glu].isna(), np.nan)

        for col_env in labels:
            sr_event = df_case_labeled[col_env]
            for n_lookahead in n_lookaheads:  # lookahead n time points
                temp = sr_event[::-1].rolling(n_lookahead, 1, closed='left').sum()[::-1]
                temp = (temp > 0).astype('int')
                col_target = f'label_{col_env}_h{n_lookahead}'
                df_case_labeled[col_target] = temp

                # at the end , the length of time is at least greater than the half
                df_case_labeled[col_target] = df_case_labeled[col_target].mask(
                    df_case_labeled['step'] > (len(df_case_labeled) - n_lookahead // 2), np.nan)

        return self._std_df_output(df_case_labeled, output_path)

    def workflow2_merge_ts(self, input_dir, output_path,
                           drug_meta_path, insulin_meta_path, test_meta_path, feat_meta_path,
                           pipeline=['base', 'sym', 'diag', 'drug', 'insulin', 'glu', 'test'],
                           skip_empty=False, mapping=True,
                           ):
        try:
            df_sample = self.workflow1_merge(input_dir, None, drug_meta_path, insulin_meta_path, test_meta_path,
                                             mapping=mapping)
            # print(df_sample)
            df_sample = self.workflow2_ts(df_sample, None, feat_meta_path)
            if skip_empty and (len(df_sample) == 0):
                output_path = 'skip'
            return self._std_df_output(df_sample, output_path, with_suffix='.csv')
        except Exception as e:
            print(input_dir)
            print(e)
            return None

    def workflow_total(self, input_dir, output_dir, meta_dir):  # for debug
        meta_dir = Path(meta_dir)
        # drug_meta_path = meta_dir / '20210607 list of hypoglycemic drugs.xlsx'
        drug_meta_path = meta_dir / '20210607降糖药列表.xlsx'
        # insulin_meta_path = meta_dir / '20220216 insulin classification.xlsx'
        insulin_meta_path = meta_dir / '20220216胰岛素分类类别.xlsx'
        # test_meta_path = meta_dir / '20220117 inspection indicators mapping table.xlsx'
        test_meta_path = meta_dir / '20220117检验指标映射表.xlsx'
        feat_meta_path = meta_dir / 'task.columns.csv'

        pipeline = ['base', 'sym', 'diag', 'drug', 'insulin', 'glu', 'test'],
        skip_empty = False

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path1 = output_dir / 'data.1.1.merge.csv'
        output_path2 = output_dir / 'data.2.1.ts.csv'
        output_path3 = output_dir / 'data.3.1.preprocess.csv'
        output_path4 = output_dir / 'data.4.1.label.csv'
        self.workflow1_merge(input_dir, output_path1, drug_meta_path, insulin_meta_path, test_meta_path)
        self.workflow2_ts(output_path1, output_path2, feat_meta_path)
        self.workflow31_preprocess(output_path2, output_path3, feat_meta_path, add_mask=True, add_timegroup=True,
                                   col_timegroup='timegroup', n_timegroup=7)
        df_temp = self.pipeline32_reward(output_path3, None)
        self.pipeline32_action(df_temp, output_path4)

    def batch(self, job, data_dir, output_root, skip_exist=False, filelist=None, file_suffix='', hash_split=None,
              n_splits=None, **kargs):
        '''
        batch processing
        Args:
        --------------------
        job: str
            task name, such as resize, rgb2gray
        data_dir: str
            directory to be processed
        output_root: str
            output path
        skip_exist: bool (default=False)
            whether to skip it if the file already exists
        num_workers: int (default=1)
            multiprocess processing (- 1 represents the use of all cores)
        '''
        func = {
            'workflow1_merge': self.workflow1_merge,
            'workflow2_ts': self.workflow2_ts,
            'workflow2_merge_ts': self.workflow2_merge_ts,
            'workflow31_preprocess': self.workflow31_preprocess,
        }[job]

        if job in ['workflow2_ts', 'workflow31_preprocess']:
            mode = 'file_input'  # input is a file 
        else:
            mode = 'dir_input'  # input is a folder

        data_dir = Path(str(data_dir))
        output_root = Path(str(output_root))
        output_root.mkdir(parents=True, exist_ok=True)

        self._data_dir = data_dir
        self._output_dir = output_root
        self._is_batch_mode = True

        from multiprocessing import Pool
        num_workers = self._num_workers
        if num_workers < 0:
            num_workers = None

        tasks = []
        if mode == 'file_input':  # expand file
            if filelist is None:
                paths = data_dir.glob(f'**/*{file_suffix}')
                paths = list(paths)
            else:
                paths = Path(filelist).read_text().strip().splitlines()
                paths = [data_dir / Path(p) for p in paths]
            for path in tqdm(paths):
                if path.is_dir():
                    continue
                filename = path.name
                if n_splits is None:
                    pass
                else:
                    hashcode = hash_func(filename)
                    if hash_split != (hashcode % n_splits):
                        continue
                output_dir = output_root / path.relative_to(data_dir)
                output_dir = output_dir.parent / filename

                if skip_exist and output_dir.exists():
                    continue
                task = (func, path, output_dir, kargs)
                tasks += [task]
        elif mode == 'dir_input':  # directory to file
            if filelist is None:
                paths = data_dir.glob(f'**')
                paths = list(paths)
            else:
                paths = Path(filelist).read_text().strip().splitlines()
                paths = [data_dir / Path(p) for p in paths]

            for path in tqdm(paths):
                if not path.is_dir():
                    continue
                dirname = path.name
                if n_splits is None:
                    pass
                else:
                    hashcode = hash_func(dirname)
                    if hash_split != (hashcode % n_splits):
                        continue
                output_path = output_root / path.relative_to(data_dir)
                output_path = output_path.with_suffix(file_suffix)

                if skip_exist and output_path.exists():
                    continue
                task = (func, path, output_path, kargs)
                tasks += [task]

        with Pool(num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(func_args_output_helper, tasks), total=len(tasks)):
                pass


if __name__ == '__main__':
    fire.Fire(DiabetesPipeline)
