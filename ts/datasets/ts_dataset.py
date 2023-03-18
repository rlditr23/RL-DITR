from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import shutil

from pathlib import Path
from functools import lru_cache
import diskcache

from .rl import reward2return
from .rl import glu2reward
from .pipe import DiabetesPipeline


def df2map(df, col_key, col_val):
    return df.drop_duplicates(col_key).set_index(col_key)[col_val]


@lru_cache(100)
def _load_df(path, col_id):
    df = pd.read_csv(path, dtype={col_id: 'str'})
    return df


@lru_cache(100)
def _get_pipeline():
    pipeline = DiabetesPipeline()
    return pipeline


class TimeseriesExtractor(object):
    """
    TimeseriesExtractor: get all time categories
    """

    def __init__(self, df_meta_path, add_mask=False, add_timegroup=False,
                 col_timegroup='timegroup', n_timegroup=7):
        """
        df_meta_path: original meta information path
        add_mask: whether to add mask features
        add_timegroup: whether to add timegroup features
        col_timegroup: the column name of timegroup
        n_timegroup: the number of time groups
        """
        super(TimeseriesExtractor, self).__init__()
        self.df_meta_path = df_meta_path
        self.add_mask = add_mask
        self.add_timegroup = add_timegroup
        self.col_timegroup = col_timegroup
        self.n_timegroup = n_timegroup
        self.pipeline = _get_pipeline()

    @property
    def n_features(self):
        return len(self.feature_names)

    @property
    @lru_cache(1)
    def feature_names(self):
        """
        obtain category information for all timegroups
        """
        df_dummy = pd.DataFrame({'timegroup': []})
        df_dummy = self.transform(df_dummy)
        return list(df_dummy.columns)

    def transform(self, df_case):
        """
        expand df_case by feature
        """
        df_output = self.pipeline.workflow31_preprocess(df_case, None, self.df_meta_path,
                                                        add_mask=self.add_mask,
                                                        add_timegroup=self.add_timegroup,
                                                        col_timegroup=self.col_timegroup,
                                                        n_timegroup=self.n_timegroup,
                                                        )
        return df_output


class ActionExtractor(object):
    """
    extract and complete the category and medication information of insulin
    """

    def __init__(self, col_insulin_dose='insulin|insulin', insulin_max=49,
                 col_insulin_cat='insulin|insulin_group', option_max=4):
        super(ActionExtractor, self).__init__()
        self.col_insulin_dose = col_insulin_dose
        self.col_insulin_cat = col_insulin_cat
        self.insulin_max = insulin_max
        self.option_max = option_max
        self.pipeline = _get_pipeline()

    def transform(self, df_case):
        df_output = self.pipeline.pipeline32_action(df_case, None,
                                                    col_insulin_dose=self.col_insulin_dose,
                                                    insulin_max=self.insulin_max,
                                                    col_insulin_cat=self.col_insulin_cat, option_max=self.option_max,
                                                    )
        return df_output


class TSRLDataset(Dataset):
    def __init__(self, df_path, df_meta_path, data_dir,
                 task='clf', cols_label=[],
                 col_id='path', col_timegroup='timegroup', n_timegroup=7,
                 max_seq_len=128,
                 n_reward_max=1, n_value_max=20,
                 feat_append_mask=False, feat_append_time=False,
                 brl_setting=False, gamma=0.9, reward_dtype='risk',
                 return_y=True,
                 cache_dir=None,
                 cache_renew=False,
                 single_df=False,
                 verbose=False,

                 n_cols_label=None,

                 # for experiments
                 outpatient_drop_ratio=None, # drop ratio 0-1 for outpatient experiments
                 noise_ratio=None, # noise ratio 0-1 for low quality data experiments
                 missing_ratio=None, # drop ratio 0-1 for missing data experiments
                 glu_missing_ratio=None, # drop ratio 0-1 for glucose missing data experiments
                 ):
        super(TSRLDataset, self).__init__()
        if not (data_dir is None):
            data_dir = Path(data_dir)

        # load the full dataframe
        if isinstance(df_path, pd.DataFrame):
            self.df = df_path
        else:
            self.df = pd.read_csv(df_path)

        if n_cols_label is not None:
            cols_label = [f'label_{i}' for i in range(n_cols_label)]
        elif len(cols_label) == 0:
            cols = self.df.columns
            cols = cols[cols.str.contains('^label_')]
            cols_label = list(cols)

        feat_extractor = TimeseriesExtractor(df_meta_path,
                                             add_mask=feat_append_mask, add_timegroup=feat_append_time,
                                             col_timegroup=col_timegroup, n_timegroup=n_timegroup)
        action_extractor = ActionExtractor(col_insulin_dose='insulin|insulin', insulin_max=49,
                                           col_insulin_cat='insulin|insulin_group', option_max=4)

        # process data_dir, df_meta_path
        self.feat_extractor = feat_extractor
        self.action_extractor = action_extractor
        self.cols_label = cols_label
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache_renew = cache_renew
        self.diskcache = None
        self.single_df = single_df
        self.col_id = col_id
        self.col_timegroup = col_timegroup

        self.feat_append_mask = feat_append_mask
        self.feat_append_time = feat_append_time
        self.max_seq_len = max_seq_len
        self.return_y = return_y
        self.brl_setting = brl_setting
        self.gamma = gamma  # discounted
        self.reward_dtype = reward_dtype  # discounted
        self.verbose = verbose

        self.n_reward_max = n_reward_max
        self.n_value_max = n_value_max
        self.n_action = 40
        self.n_option = 4

        # for experiments
        self.outpatient_drop_ratio = outpatient_drop_ratio
        self.noise_ratio = noise_ratio
        self.missing_ratio = missing_ratio
        self.glu_missing_ratio = glu_missing_ratio

    @property
    def n_features(self):
        return self.feat_extractor.n_features

    @property
    def n_labels(self):
        return len(self.labels)

    @property
    def feature_names(self):
        return self.feat_extractor.feature_names

    @property
    def labels(self):
        return self.cols_label

    # @lru_cache(100000)
    def _get_sample(self, path, max_seq_len, gamma, reward_dtype,
                    outpatient_drop_ratio=None, noise_ratio=None,
                    missing_ratio=None,
                    glu_missing_ratio=None):
        """

        Outputs:
        -------------------
        path: str
            example:

        df_case_mask: array[T,C], {0,1}
            mask, 1 indicates the original value
        df_case_target: array[T,C], float
        """
        gamma = float(gamma)

        # obtain a single sample with timing points
        if isinstance(path, pd.DataFrame):
            df_case_raw = path
        else:
            df_case_raw = pd.read_csv(path)

        # outpatient experiments
        if outpatient_drop_ratio is not None:
            # each lab test has one value at most
            # glucose values drop

            cols_test = df_case_raw.columns[df_case_raw.columns.str.contains(r'^test\|', regex=True)]
            for col in cols_test:
                idx = df_case_raw[col].first_valid_index()
                if idx is not None:
                    df_case_raw.loc[idx+1:, col] = np.nan

            col_glu = 'glu|glu'
            if col_glu in df_case_raw.columns:
                steps_drop = df_case_raw['step'].sample(frac=outpatient_drop_ratio)
                df_case_raw[col_glu] = df_case_raw[col_glu].mask(df_case_raw['step'].isin(steps_drop), np.nan)

        # low data quality experiments
        if noise_ratio is not None:
            # each lab test or glu has random noise scaled by the original value
            cols = df_case_raw.columns[df_case_raw.columns.str.contains(r'^test\||^glu\|', regex=True)]
            # cols = cols[:5]
            # print(df_case_raw[cols].std())
            scales = np.random.uniform(-noise_ratio, noise_ratio, size=(df_case_raw.shape[0], len(cols)))
            df_case_raw[cols] = df_case_raw[cols] * (1 + scales)
            # print(df_case_raw[cols].mean())
            # print(df_case_raw[cols].std())

        # missing data experiments
        if missing_ratio is not None:
            # each lab test or glu has random noise scaled by the original value
            cols = df_case_raw.columns[df_case_raw.columns.str.contains(r'^test\||^glu\|', regex=True)]
            # print(df_case_raw[cols].notna().sum())
            masks = np.random.uniform(0, 1, size=(df_case_raw.shape[0], len(cols))) < missing_ratio # set to nan
            df_case_raw[cols] = df_case_raw[cols].mask(masks, np.nan)
            # print(df_case_raw[cols].notna().sum())

        # glu missing experiments
        if glu_missing_ratio is not None:
            col_glu = 'glu|glu'
            if col_glu in df_case_raw.columns:
                masks = np.random.uniform(0, 1, size=len(df_case_raw)) < glu_missing_ratio # set to nan
                df_case_raw[col_glu] = df_case_raw[col_glu].mask(masks, np.nan)

        # extract features
        df_case_feat = self.feat_extractor.transform(df_case_raw)

        # df_case -> df_x, df_y
        df_obs = df_case_feat
        df_aux = df_case_raw.reindex(columns=self.labels)
        df_mask_aux = df_aux.notna().astype('int')

        # get the value
        obs = df_obs.fillna(0).values
        aux = df_aux.fillna(0).values
        mask_aux = df_mask_aux.values

        result = {
            'obs': obs,
            'aux': aux,
            'mask_aux': mask_aux,
        }

        if self.brl_setting:
            col_glu = 'glu|glu'
            col_insulin_dose = 'insulin|insulin'
            col_insulin_cat = 'insulin|insulin_group'
            df_case_raw['reward'] = glu2reward(df_case_feat[col_glu], dtype=reward_dtype)
            df_case_raw['return'] = reward2return(df_case_raw['reward'], gamma=gamma)
            df_case_raw = self.action_extractor.transform(df_case_raw) # add action and option

            df_reward = df_case_raw['reward']
            df_return = df_case_raw['return']
            df_action = df_case_raw['action']
            df_option = df_case_raw['option']

            reward = df_reward.fillna(0).values
            cumreward = df_return.fillna(0).values
            action = df_action.fillna(0).values
            option = df_option.fillna(0).values

            df_mask_action = (df_action > 0).astype('int')
            mask_action = df_mask_action.values

            glu = df_case_feat[col_glu].values

            result.update({
                'reward': reward,
                'cumreward': cumreward,
                'action': action,
                'option': option,
                'mask_action': mask_action,
                'glu': glu,
            })

        # padding: [:l] is False, [l:] is True
        l = len(obs)
        padding = np.zeros(max_seq_len, dtype='bool')
        if l < max_seq_len:
            delta = max_seq_len - l
            for key, val in result.items():
                if len(val.shape) == 1:
                    result[key] = np.pad(val, (0, delta), 'constant', constant_values=0)
                elif len(val.shape) == 2:
                    result[key] = np.pad(val, ((0, delta), (0, 0)), 'constant', constant_values=0)
            padding[l:] = True
        else:  # cut off
            for key, val in result.items():
                result[key] = val[:max_seq_len]
        result.update({
            'padding': padding,
        })

        return result

    def __len__(self):
        if self.single_df:
            return 1
        else:
            return len(self.df)

    def __getitem__(self, idx):
        """
        Outputs:
        x: float[L,D]
        y: float[L,Y]
        x_padding: bool[L,D], True indicates that it is padded (should be ignored by attention)
        y_padding: bool[L,Y]
        x_mask: int[L,D], 1 indicates that it is vaild
        y_mask: int[L,Y], 1 indicates that it is vaild

        NOTE:
        - composition of x features
            - original features (M)
            - validity of original features (M) 01 mask
            - time of day (7) indictor
            - predict target lookahead (Y, Optional)
        """

        if not (self.cache_dir is None):
            if self.diskcache is None:
                if self.cache_renew:
                    shutil.rmtree(self.cache_dir, ignore_errors=True)
                self.diskcache = diskcache.Cache(directory=self.cache_dir, eviction_policy='none')

        if self.single_df:
            key = self.df, self.max_seq_len, f'{self.gamma:.4f}', self.reward_dtype, \
                  self.outpatient_drop_ratio, self.noise_ratio, self.missing_ratio, self.glu_missing_ratio
            sample = self._get_sample(*key)
        else:
            case_meta = self.df.iloc[idx]
            path = self.data_dir / case_meta['path']
            key = path, self.max_seq_len, f'{self.gamma:.4f}', self.reward_dtype, \
                  self.outpatient_drop_ratio, self.noise_ratio, self.missing_ratio, self.glu_missing_ratio
            if (self.diskcache is None):
                sample = self._get_sample(*key)
            else:
                if key in self.diskcache:
                    sample = self.diskcache[key]
                else:
                    sample = self._get_sample(*key)
                    self.diskcache[key] = sample

        # load the obtained data without any offset
        # only handle label
        obs = sample['obs'].astype('float32')
        aux = sample['aux'].astype('float32')  # for BCE loss
        aux = np.roll(aux, -1, axis=0)
        aux[-1] = 0
        mask_aux = sample['mask_aux'].astype('int')
        mask_aux = np.roll(mask_aux, -1, axis=0)
        mask_aux[-1] = 0
        padding = sample['padding'].astype('bool')
        glu = sample['glu'].astype('float32')
        glu_target = np.roll(glu, -1)  # move forward one point of time as a label
        glu_target[-1] = -100  # the last reward is 0

        if not self.brl_setting:
            if self.return_y:
                return (obs, padding), (aux, mask_aux)
            else:
                return (obs, padding)
        else:
            reward = sample['reward'].astype('float32')
            reward = np.roll(reward, -1)  # move forward one point of time as a label
            reward[-1] = 0  # the last reward is 0

            cumreward = sample['cumreward'].astype('float32')
            cumreward = np.roll(cumreward, -1)  # move forward one point of time as a label
            cumreward[-1] = 0  # the last reward is 0

            action = sample['action'].astype('int')
            action = action.clip(0, self.n_action - 1)
            action_prev = np.roll(action, shift=1)
            action_prev[0] = 0
            option = sample['option'].astype('int')
            option = option.clip(0, 4)
            mask_reward = glu > 0.01
            mask_reward[-1] = False

            if self.return_y:
                label_data = {
                    'aux': aux,
                    'mask_aux': mask_aux,
                    'reward': reward,
                    'cumreward': cumreward,
                    'action': action,
                    'mask_reward': mask_reward,
                    'glu_target': glu_target,
                    'padding': padding,
                }
                return (obs, action_prev, option, padding), label_data
            else:
                return (obs, action_prev, option, padding)
