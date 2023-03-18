import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

if (__name__ == '__main__') or (__package__ == ''):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
from ts.datasets.pipe import DiabetesPipeline
from ts.datasets.ts_dataset import TSRLDataset
from ts.models.agent import InsulinArmAgent
# else:
#     from .datasets.pipe import DiabetesPipeline
#     from .datasets.ts_dataset import TSRLDataset
#     from .models.agent import InsulinArmAgent


from torch.utils.data.dataloader import default_collate


def dataframe2tensor(df_case, size, df_meta_path):
    '''
    convert dataframe to tensor
    see TSLRDataset for more details
    '''
    data_dir = None

    dataset_kargs = {}
    dataset_kargs_ori = {
        'max_seq_len': size,
        'brl_setting': True,
        'feat_append_mask': True,
        'feat_append_time': True,
        'return_y': False,
    }

    dataset_kargs_ori.update(dataset_kargs)
    dataset_kargs_ori['single_df'] = True
    ds_single = TSRLDataset(df_case, df_meta_path, data_dir, **dataset_kargs_ori)
    result = ds_single[0]
    result = default_collate([result])
    return result


class Model(object):
    def __init__(self, model_dir, df_meta_path, beam_size=2, device=None):
        super(Model, self).__init__()
        self.df_meta_path = df_meta_path
        self.beam_size = beam_size
        self.model_dir = model_dir

        self.pipeline = DiabetesPipeline()
        self.agent = InsulinArmAgent.build_from_dir(model_dir, device)
        self.size = self.agent.max_len

    def find_first_timepoint(self, sr_dt):
        """
        Find the first NA index
        """
        index_s = sr_dt.isna().tolist().index(False)
        t_norm = [
            pd.Timedelta("6:00:00"),
            pd.Timedelta("8:30:00"),
            pd.Timedelta("10:30:00"),
            pd.Timedelta("13:00:00"),
            pd.Timedelta("16:30:00"),
            pd.Timedelta("19:00:00"),
            pd.Timedelta("21:00:00")
        ]
        t_one_day = pd.Timedelta("24:00:00")
        # Record the first NA date index in t_norm
        record_index = 0
        # Record the first NA date
        record_day = 0
        for i in range(len(t_norm)):
            if (sr_dt[index_s] - t_norm[i]).hour == 0:
                record_index = i
                record_day = sr_dt[index_s] - t_norm[i]
        for i in range(len(sr_dt)):
            sr_dt[i] = record_day + t_norm[(i - index_s + record_index) % 7] + t_one_day * (
                        (i - index_s + record_index) // 7)
        return sr_dt

    def _parse_dataframe(self, df, scheme, start_time, days):
        """
        parse input tables
        """
        if isinstance(df, str) or isinstance(df, Path):
            df = pd.read_csv(df)

        df['datetime'] = pd.to_datetime(df['datetime'])

        # add prediction request
        df1ext = df.copy()
        df1ext = self.pipeline.pipeline2ext_add_option(df1ext, None, scheme, start_time, days)
        tmax = df1ext['datetime'].max()
        tmin = tmax - pd.Timedelta(17, unit='D')  # cut off
        df1ext = df1ext[df1ext['datetime'] > tmin]

        # converted data table to time series characteristics
        df2 = self.pipeline.workflow2_ts(df1ext, None, self.df_meta_path)
        return df2

    def predict(self, df, scheme, start_time, days, beam_size=None):
        # parse input tables
        df2 = self._parse_dataframe(df, scheme, start_time, days + 1)  # extra one day

        # convert to tensor
        tensors = dataframe2tensor(df2, size=self.size, df_meta_path=self.df_meta_path)
        obs, action, option, padding = tensors

        # time point to order index
        t_start = pd.Timestamp(start_time)
        t_end = t_start + pd.Timedelta(days, unit='D')

        sr_dt = df2['datetime_norm'].copy()
        sr_dt = self.find_first_timepoint(sr_dt)
        df2['datetime_norm'] = sr_dt
        t0 = sr_dt.searchsorted(t_start)
        tt = min(t0 + days * 7, self.size)
        tt = min(tt, len(df2))

        # predict
        beam_size = beam_size if beam_size is not None else self.beam_size
        output = self.agent.self_rollout(obs, action, option, t0, tt + 7, beam_size=beam_size)  # extra one day
        actions_out, options_out, values_out, rewards_out = output
        # print(df2[['datetime_norm','insulin|insulin','glu|glu']].iloc[t0-7:tt+1])

        # convert to output format
        ts = sr_dt.iloc[t0:tt]
        df_result = ts.to_frame('datetime')
        df_result['action'] = actions_out[:-7]  # remove extra one day
        df_result['option'] = options_out[:-7]  # remove extra one day

        df_output = df_result[df_result['option'] > 0]
        df_output = df_output.drop(columns=['option'])
        df_output['datetime'] = df_output['datetime'].astype('str')
        df_output = df_output.rename(columns={'action': 'dose'})
        result = df_output.to_dict('records')

        return result


def predict(model_dir, df_meta_path, csv_path, scheme, start_time, days=1, beam_size=5):
    df_data = pd.read_csv(csv_path)
    df_data['datetime'] = pd.to_datetime(df_data['datetime'])
    model = Model(model_dir=model_dir, df_meta_path=df_meta_path)
    result = model.predict(df_data, scheme, start_time, days, beam_size=beam_size)
    return result


if __name__ == '__main__':
    import fire

    # df_meta_path = 'assets/models/features.csv'
    # model_dir = 'assets/models/weights'
    # csv_path = 'assets/data/sample.csv'
    # scheme = ['premixed', 'na', 'premixed', 'na']
    # start_time = '2022-01-16'
    # days = 2
    result = fire.Fire(predict)
