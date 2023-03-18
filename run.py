#!/usr/bin/env python3
import json
import hashlib

import fire
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer

from light.core.logger import ConsoleLogger
from ts.pl_module.base import BaseModule
from ts.pl_module.embedding import DiabetesRLSLTestEmbeddingModelModule
from ts.pl_module.patient_module import DiabetesPatientModelModule
from ts.utils import support_to_scalar, logit_regression_loss, logit_regression_mae, masked_loss, masked_mean


def hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def split_data_by_col(sr: pd.Series, split_i: int, split_n: int, split_salt: str = ''):
    sr = sr.reset_index(drop=True)
    sr_hash = (sr.astype('str') + split_salt).apply(hash)
    sr = sr[sr_hash % split_n == split_i]
    return sr


class DiabetesDataModule(pl.LightningDataModule):
    def __init__(self, df_path, task, data_args=None, col_group='dataset', batch_size=32, num_workers=0, pin_memory=True, shuffle=True):
        if data_args is None:
            data_args = {}
        super().__init__()
        self.df_path = df_path
        self.task = task
        self.data_args = data_args
        self.col_group = col_group
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        from ts.datasets.ts_dataset import TSRLDataset
        data_kargs_default = {
            'df_meta_path': 'meta/task.columns.csv',
            'max_seq_len': 128,
            'brl_setting': True,
            'reward_dtype': 'risk',
            'return_y': True,
            'feat_append_mask': True,
            'feat_append_time': True,
        }
        data_kargs_default.update(self.data_args)

        df = pd.read_csv(self.df_path)
        df_train = df[df[self.col_group].isin(['train'])]
        df_valid = df[df[self.col_group].isin(['valid'])]
        df_test = df[df[self.col_group].isin(['valid', 'test', 'other'])]

        self.ds_train = TSRLDataset(df_train, **data_kargs_default)
        self.ds_valid = TSRLDataset(df_valid, **data_kargs_default)
        self.ds_test = TSRLDataset(df_test, **data_kargs_default)

        self.n_input = self.ds_train.n_features
        self.n_labels = self.ds_train.n_labels
        self.max_length = self.ds_train.max_seq_len
        self.n_reward_max = self.ds_train.n_reward_max
        self.n_value_max = self.ds_train.n_value_max
        self.n_action = self.ds_train.n_action
        self.n_option = self.ds_train.n_option

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False, )

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False)

    def teardown(self, stage=None):
        pass


def get_model(task, model_name='tf', model_path=None, model_args={}, data_module=None):
    n_input = data_module.n_input
    n_labels = data_module.n_labels
    max_length = data_module.max_length
    n_reward_max = data_module.n_reward_max
    n_value_max = data_module.n_value_max
    n_action = data_module.n_action
    n_option = data_module.n_option

    model = None
    if model_path is None:  # create a new model
        if task == 'rlsl':
            from ts.models.rlsl_model import TransformerPlanningModel
            model_kargs_ori = {}
            model_kargs_ori.update({
                'n_input': n_input,
                'n_aux': n_labels,
                'max_len': max_length,
                'n_action': n_action,
                'n_option': n_option,
                'n_reward_max': n_reward_max,
                'n_value_max': n_value_max,
            })
            model_kargs_ori.update(model_args)
            model = TransformerPlanningModel(**model_kargs_ori)
        elif task == 'sl':
            model_args_t = model_args.copy()
            n_step = model_args_t.pop('n_step', 1)
            n_inputs = n_input + n_action + n_option
            n_outputs = (n_action + n_labels + 1) * n_step
            if model_name == 'lstm':
                from ts.models.baseline import LSTM
                model_kargs_ori = {
                    'n_input': n_inputs,
                    'n_outputs': n_outputs,
                    'max_len': max_length,
                }
                model_kargs_ori.update(model_args_t)
                model = LSTM(**model_kargs_ori)
            elif model_name == 'tf':
                from ts.models.baseline import TransformerModel
                model_kargs_ori = {
                    'n_input': n_inputs,
                    'n_outputs': n_outputs,
                    'max_len': max_length,
                }
                model_kargs_ori.update(model_args_t)
                model = TransformerModel(**model_kargs_ori)
            elif model_name == 'cnn':
                from ts.models.baseline import CNN
                model_kargs_ori = {
                    'input_size': n_inputs,
                    'output_size': n_outputs,
                    'layer_sizes': [256] * 3,
                }
                model_kargs_ori.update(model_args_t)
                model = CNN(**model_kargs_ori)
            elif model_name == 'mlp':
                from ts.models.baseline import MLP
                model_kargs_ori = {
                    'input_size': n_inputs,
                    'output_size': n_outputs,
                    'layer_sizes': [256] * 3,
                }
                model_kargs_ori.update(model_args_t)
                model = MLP(**model_kargs_ori)
        elif task == 'patient':
            from ts.models.patient_model import PatientModel
            model_kargs_ori = {
                'n_input': n_input,
                'n_aux': n_labels,
                'max_len': max_length,
                'n_reward_max': n_reward_max,
                'n_value_max': n_value_max,
                'n_step': 1,
            }
            model_kargs_ori.update(model_args)
            model = PatientModel(**model_kargs_ori)
    else:
        model = torch.load(model_path)
    return model


class DiabetesRLSLModelModule(BaseModule):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.loss_state_weight = self.module_args.get('loss_state_weight', 0.1)
        self.loss_rl_weight = self.module_args.get('loss_rl_weight', 0)
        self.loss_joint = self.module_args.get('loss_rl_joint', False)

    def forward(self, x):
        out = self.model(*x)
        return out

    def get_loss(self, pred_data, label_data):
        action = label_data['action']
        mask_reward = label_data['mask_reward']  # (b, l)
        aux = label_data['aux']
        glu_target = label_data['glu_target']
        mask_aux = label_data['mask_aux']
        reward = label_data['reward']
        cumreward = label_data['cumreward']
        padding = label_data['padding']

        pred_policy = pred_data['policy']
        pred_policy_train = pred_data['policy_train']
        pred_aux = pred_data['aux']
        pred_glu = pred_data['glu']
        pred_state = pred_data['state']
        pred_value = pred_data['value']
        pred_reward = pred_data['reward']

        b, n_step, length = pred_policy.shape[:3]
        device = pred_policy.device

        # mask
        step_mask = torch.ones([length], device=device)  # (batch, )
        step_masks = [step_mask]
        for i in range(n_step - 1):
            step_mask = torch.cat([torch.zeros_like(step_mask[-1:]), step_mask[:-1]])
            step_masks += [step_mask]
        step_mask = torch.stack(step_masks)  # [n_step, len]
        step_mask = step_mask[None, ...].expand(b, n_step, length)  # (b, n, l)
        step_mask = step_mask.bool()

        # action function
        pred_action = pred_policy
        pred_action_train = pred_policy_train
        n_action = pred_action_train.shape[-1]
        true_action = action.expand_as(pred_action_train[..., 0]).clone()
        true_action_mask = true_action > 0
        true_action_mask = true_action_mask * step_mask
        true_action[~true_action_mask] = -100

        loss_action = F.cross_entropy(pred_action_train.reshape(-1, n_action), true_action.reshape(-1),
                                      ignore_index=-100)
        if not self.loss_joint:
            loss_reward = 0
            loss_aux = 0
            loss_glu = 0
            loss_state = 0
            loss = loss_action
        else:
            # value function
            if pred_value.size(-1) > 1:
                loss_value = logit_regression_loss(pred_value, cumreward, mask=step_mask)
            else:
                loss_value = masked_loss(F.mse_loss, pred_value.squeeze(-1), cumreward, mask=step_mask)

            # reward function
            mask_reward = mask_reward.expand(b, n_step, length).bool()
            if pred_reward.size(-1) > 1:
                loss_reward = logit_regression_loss(pred_reward, reward, mask=step_mask * mask_reward)
            else:
                loss_reward = masked_loss(F.mse_loss, pred_reward.squeeze(-1), reward,
                                          mask=step_mask * mask_reward)

            # state loss function
            loss_state_all = 0
            mask0 = padding
            for i in range(n_step - 1):
                state_shift = torch.roll(pred_state[:, i + 1], shifts=i + 1, dims=1)
                mask_shift = torch.roll(mask0, shifts=i + 1, dims=1)
                mask_shift[:, :i + 1] = 0
                loss_state_t = torch.sqrt((pred_state[:, 0] - state_shift) ** 2 + 1e-8).mean(dim=-1)
                mask_t = mask0 * mask_shift
                loss_state_all += (loss_state_t * mask_t).sum() / mask_t.sum().clip(1)
            loss_state = loss_state_all / max(n_step - 1, 1)

            # glu loss
            glu_mask = glu_target > 0.1
            glu_mask = glu_mask.expand(b, n_step, length).bool()
            loss_glu = masked_loss(F.mse_loss, pred_glu.squeeze(-1), glu_target, mask=step_mask * glu_mask)

            # glu event loss
            n_out = aux.shape[-1]
            pred_aux = pred_aux
            true_aux = aux.expand_as(pred_aux).clone()
            loss_aux_all = F.binary_cross_entropy_with_logits(pred_aux.reshape(-1, n_out), true_aux.reshape(-1, n_out),
                                                              reduction='none')
            mask = (mask_aux * step_mask[..., None]).reshape(-1, n_out)
            loss_aux = (loss_aux_all * mask).sum() / mask.sum().clip(1)

            loss = loss_action + 0.5 * loss_aux + 0.1 * loss_glu + 0.05 * loss_value + 0.05 * loss_reward + self.loss_state_weight * loss_state
        losses = {
            '_step_mask': step_mask,
            '_mask_reward': mask_reward,
            '_glu_mask': glu_mask,
            '_pred_action': pred_action,
            '_true_action': true_action,

            'loss': loss,
            'loss_action': loss_action,
            'loss_value': loss_value,
            'loss_reward': loss_reward,
            'loss_aux': loss_aux,
            'loss_glu': loss_glu,
            'loss_state': loss_state,
        }

        if 'policy_return' in pred_data:
            policy_action_logprob = pred_data['policy_action_logprob']
            policy_return = support_to_scalar(pred_data['policy_return'])
            loss_rl = masked_mean(-policy_action_logprob * policy_return.detach(), mask=step_mask)
            loss_rl = torch.clip(loss_rl, -1, 1)
            losses['loss_rl'] = loss_rl
            losses['loss'] += self.loss_rl_weight * loss_rl
        return losses

    def training_step(self, batch, batch_idx):
        x, y = batch
        label_data = y
        pred_data = self.model(*x, sample=True)
        b, n_step, length = pred_data['policy'].shape[:3]
        label_data, stack_mask = self.label_data_stack(label_data, n_step)
        losses = self.get_loss(pred_data, label_data)
        loss = losses['loss']
        self.train_loss = loss.detach()
        return loss

    def validation_step(self, batch, batch_idx):
        x, label_data = batch
        pred_data = self.model(*x)
        b, n_step, length = pred_data['policy'].shape[:3]
        label_data, stack_mask = self.label_data_stack(label_data, n_step)
        losses = self.get_loss(pred_data, label_data)
        loss = losses['loss']

        if self.train_loss is None:
            self.train_loss = 0
        self.log("train_loss", self.train_loss, prog_bar=False, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)

        action = label_data['action']
        aux = label_data['aux']
        mask_aux = label_data['mask_aux']
        glu_target = label_data['glu_target']
        reward = label_data['reward']
        mask_reward = label_data['mask_reward']  # (b, l)
        cumreward = label_data['cumreward']

        pred_policy = pred_data['policy']
        pred_aux = pred_data['aux']
        pred_glu = pred_data['glu']
        pred_state = pred_data['state']
        pred_value = pred_data['value']
        pred_reward = pred_data['reward']
        b, n_step, length = pred_policy.shape[:3]
        device = pred_policy.device

        # mask
        step_mask = losses['_step_mask']
        glu_mask = losses['_glu_mask']
        mask_reward = losses['_mask_reward']

        # action function
        pred_action = losses['_pred_action']
        true_action = losses['_true_action']

        # metrics action
        pred_action_scalar = pred_action.argmax(dim=-1)
        action_mae_all = torch.abs(pred_action_scalar - true_action)
        valid = true_action.expand_as(action_mae_all) > 0
        action_mae = (action_mae_all * valid).sum() / valid.sum().clip(1)

        # metrics value mae
        if pred_value.size(-1) > 1:
            value_mae = logit_regression_mae(pred_value, cumreward, mask=step_mask)
        else:
            value_mae = masked_loss(F.l1_loss, pred_value.squeeze(-1), cumreward, mask=step_mask)

        # metrics reward mae
        if pred_reward.size(-1) > 1:
            reward_mae = logit_regression_mae(pred_reward, reward, mask=step_mask * mask_reward)
        else:
            reward_mae = masked_loss(F.l1_loss, pred_reward.squeeze(-1), reward, mask=step_mask * mask_reward)

        glu_mae = masked_loss(F.l1_loss, pred_glu.squeeze(-1), glu_target, mask=step_mask * glu_mask)

        name_to_value = {
            'loss_action': losses['loss_action'],
            'loss_value': losses['loss_value'],
            'loss_reward': losses['loss_reward'],
            'loss_aux': losses['loss_aux'],
            'loss_glu': losses['loss_glu'],
            'loss_state': losses['loss_state'],
            'action_mae': action_mae,
            'value_mae': value_mae,
            'reward_mae': reward_mae,
            'glu_mae': glu_mae,
        }
        if 'loss_rl' in losses:
            name_to_value['loss_rl'] = losses['loss_rl']

        for metrics_name, value in name_to_value.items():
            metrics = self.metrics[metrics_name]
            metrics.update(value)
            metrics_attr_name = f"val_{metrics_name}"
            self.log(metrics_attr_name, metrics, prog_bar=False, sync_dist=True, metric_attribute=metrics_attr_name)

        return loss

    def test_step(self, batch, batch_idx):
        x, label_data = batch
        n_step = self.n_step
        label_data, stack_mask = self.label_data_stack(label_data, n_step=n_step)
        pred_data = self.model(*x)
        obs_x, action_x, option_x, padding_x = x
        pred_policy = pred_data['policy']
        pred_value = pred_data['value']
        pred_reward = pred_data['reward']
        pred_aux = pred_data['aux']
        pred_glu = pred_data['glu']

        action = label_data['action']
        aux = label_data['aux']
        glu_target = label_data['glu_target']
        mask_aux = label_data['mask_aux']
        reward = label_data['reward']
        cumreward = label_data['cumreward']

        pred_policy = F.softmax(pred_policy, dim=-1)
        pred_aux = torch.sigmoid(pred_aux)
        pred_value = support_to_scalar(pred_value)
        pred_reward = support_to_scalar(pred_reward)
        pred_action = pred_policy.argmax(dim=-1)

        result = {}
        result['padding_mask'] = padding_x.to('cpu')
        result['stack_mask'] = stack_mask.to('cpu')

        result['pred_policy'] = pred_policy.to('cpu')
        result['pred_action'] = pred_action.to('cpu')
        result['pred_aux'] = pred_aux.to('cpu')
        result['pred_glu'] = pred_glu.to('cpu')
        result['pred_reward'] = pred_reward.to('cpu')
        result['pred_value'] = pred_value.to('cpu')

        result['action'] = action.to('cpu')
        result['option'] = option_x.to('cpu')
        result['aux'] = aux.to('cpu')
        result['mask_aux'] = mask_aux.to('cpu')
        result['glu_target'] = glu_target.to('cpu')
        result['reward'] = reward.to('cpu')
        result['cumreward'] = cumreward.to('cpu')

        return result


class DiabetesSLModelModule(BaseModule):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        self.n_action = self.data_module.n_action
        self.n_option = self.data_module.n_option
        self.n_labels = self.data_module.n_labels
        self.n_outputs = self.n_action + self.n_labels + 1

    def forward(self, x):
        obs, action_prev, option, padding = x
        action_prev = F.one_hot(action_prev, num_classes=self.n_action)
        option = F.one_hot(option, num_classes=self.n_option)
        x = torch.cat([obs, action_prev, option], dim=-1)
        out = self.model(x, padding)
        b, l, ho = out.shape

        out = out.reshape(b, l, -1, self.n_outputs).permute(0, 2, 1, 3)
        output = {
            'policy': out[..., :self.n_action],
            'aux': out[..., self.n_action:self.n_action + self.n_labels],
            'glu': out[..., -1:],
        }
        return output

    def get_loss(self, pred_data, label_data, stack_mask=None):
        action = label_data['action']
        aux = label_data['aux']
        glu_target = label_data['glu_target']
        mask_aux = label_data['mask_aux']

        pred_policy = pred_data['policy']
        pred_aux = pred_data['aux']
        pred_glu = pred_data['glu']

        b, n_step, length = pred_policy.shape[:3]
        device = pred_policy.device

        # mask
        step_mask = torch.ones([length], device=device)  # (batch, )
        step_masks = [step_mask]
        for i in range(n_step - 1):
            step_mask = torch.cat([torch.zeros_like(step_mask[-1:]), step_mask[:-1]])
            step_masks += [step_mask]
        step_mask = torch.stack(step_masks)  # [n_step, len]
        step_mask = step_mask[None, ...].expand(b, n_step, length)  # (b, n, l)
        step_mask = step_mask.bool()

        # action loss
        pred_action = pred_policy
        n_action = pred_policy.shape[-1]
        true_action = action.expand_as(pred_action[..., 0]).clone()
        true_action_mask = true_action > 0
        true_action_mask = true_action_mask * step_mask
        true_action[~true_action_mask] = -100
        loss_action = F.cross_entropy(pred_action.reshape(-1, n_action), true_action.reshape(-1), ignore_index=-100)

        # glu loss
        glu_mask = glu_target > 0
        glu_mask = glu_mask.expand(b, n_step, length).bool()
        loss_glu = masked_loss(F.mse_loss, pred_glu.squeeze(-1), glu_target, mask=step_mask * glu_mask)

        # glu event loss
        n_out = aux.shape[-1]
        pred_aux = pred_aux
        true_aux = aux.expand_as(pred_aux).clone()
        loss_aux_all = F.binary_cross_entropy_with_logits(pred_aux.reshape(-1, n_out), true_aux.reshape(-1, n_out),
                                                          reduction='none')
        mask = (mask_aux * step_mask[..., None]).reshape(-1, n_out)
        loss_aux = (loss_aux_all * mask).sum() / mask.sum().clip(1)
        loss = loss_action + 0.5 * loss_aux + 0.1 * loss_glu
        losses = {
            '_step_mask': step_mask,
            '_glu_mask': glu_mask,
            '_pred_action': pred_action,
            '_true_action': true_action,

            'loss': loss,
            'loss_action': loss_action,
            'loss_aux': loss_aux,
            'loss_glu': loss_glu,
        }
        return losses

    def training_step(self, batch, batch_idx):
        x, y = batch
        label_data = y
        pred_data = self.forward(x)
        b, n_step, length = pred_data['policy'].shape[:3]
        label_data, stack_mask = self.label_data_stack(label_data, n_step)
        losses = self.get_loss(pred_data, label_data, stack_mask=stack_mask)
        loss = losses['loss']
        self.train_loss = loss.detach()
        return loss

    def validation_step(self, batch, batch_idx):
        x, label_data = batch
        pred_data = self.forward(x)
        b, n_step, length = pred_data['policy'].shape[:3]
        label_data, stack_mask = self.label_data_stack(label_data, n_step)
        losses = self.get_loss(pred_data, label_data)
        loss = losses['loss']

        if self.train_loss is None:
            self.train_loss = 0
        self.log("train_loss", self.train_loss, prog_bar=False, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)

        action = label_data['action']
        aux = label_data['aux']
        mask_aux = label_data['mask_aux']
        glu_target = label_data['glu_target']

        pred_policy = pred_data['policy']
        pred_aux = pred_data['aux']
        pred_glu = pred_data['glu']
        b, n_step, length = pred_policy.shape[:3]
        device = pred_policy.device

        # mask
        step_mask = losses['_step_mask']
        glu_mask = losses['_glu_mask']

        # action function
        pred_action = losses['_pred_action']
        true_action = losses['_true_action']

        # metrics action
        pred_action_scalar = pred_action.argmax(dim=-1)
        action_mae_all = torch.abs(pred_action_scalar - true_action)
        valid = true_action.expand_as(action_mae_all) > 0
        action_mae = (action_mae_all * valid).sum() / valid.sum().clip(1)

        glu_mae = masked_loss(F.l1_loss, pred_glu.squeeze(-1), glu_target, mask=step_mask * glu_mask)

        name_to_value = {
            'loss_action': losses['loss_action'],
            'loss_aux': losses['loss_aux'],
            'loss_glu': losses['loss_glu'],
            'action_mae': action_mae,
            'glu_mae': glu_mae,
        }
        for metrics_name, value in name_to_value.items():
            metrics = self.metrics[metrics_name]
            metrics.update(value)
            metrics_attr_name = f"val_{metrics_name}"
            self.log(metrics_attr_name, metrics, prog_bar=False, sync_dist=True, metric_attribute=metrics_attr_name)

        return loss

    def test_step(self, batch, batch_idx):
        x, label_data = batch
        n_step = self.n_step
        label_data, stack_mask = self.label_data_stack(label_data, n_step=n_step)
        pred_data = self.forward(x)
        obs_x, action_x, option_x, padding_x = x
        pred_policy = pred_data['policy']
        pred_aux = pred_data['aux']
        pred_glu = pred_data['glu']

        aux = label_data['aux']
        glu_target = label_data['glu_target']
        mask_aux = label_data['mask_aux']
        action = label_data['action']

        pred_policy = F.softmax(pred_policy, dim=-1)
        pred_aux = torch.sigmoid(pred_aux)
        pred_action = pred_policy.argmax(dim=-1)

        result = {}
        result['padding_mask'] = padding_x.to('cpu')
        result['stack_mask'] = stack_mask.to('cpu')
        result['pred_policy'] = pred_policy.to('cpu')
        result['pred_action'] = pred_action.to('cpu')
        result['pred_aux'] = pred_aux.to('cpu')
        result['pred_glu'] = pred_glu.to('cpu')

        result['action'] = action.to('cpu')
        result['option'] = option_x.to('cpu')
        result['aux'] = aux.to('cpu')
        result['mask_aux'] = mask_aux.to('cpu')
        result['glu_target'] = glu_target.to('cpu')
        return result


class DiabetesLightRunner(object):
    def __init__(self, valid_ratio=0.2, gpus=0, batch_size=8, num_workers=0, pin_memory=True):
        super(DiabetesLightRunner, self).__init__()
        self._gpus = gpus
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._valid_ratio = valid_ratio
        self._pin_memory = pin_memory

    def train(self, task, df_path, data_root, output_dir, *, col_group='dataset', debug=False,
              model_name='tf_plan',
              shuffle=True, lr=0.001, n_epoch=10, patience=10,
              model_args={}, data_args={}, module_args={}, data_config=None,
              ):
        """
        Training

        Args:
        --------------------
        task: str
            Task Type (supports 4 types for now)
            - patient: patient model training or testing
            - rlsl: reinforcement learning with supervised learning
            - sl: supervised learning models
        df_path: str
            data table file path
            - contains at least two columns: image_path, label
            - multiple labels tasks are stored in labels
        data_root: str
            dataset root path
        output_dir: str
            result output path

        """
        output_dir = Path(output_dir)
        data_args['data_dir'] = data_root

        if data_config is not None:
            data_args_default = json.loads(Path(data_config).read_text())
            data_args_default.update(data_args)
            data_args = data_args_default

        # data module
        data_module = DiabetesDataModule(df_path, task=task, data_args=data_args, col_group=col_group,
                                         batch_size=self._batch_size, num_workers=self._num_workers,
                                         pin_memory=self._pin_memory, shuffle=shuffle,
                                         )
        data_module.setup()

        # model module
        if isinstance(model_args, str):
            try:
                model_args = eval(model_args)
            except Exception as e:
                print(e)
        if isinstance(data_args, str):
            try:
                data_args = eval(data_args)
            except Exception as e:
                print(e)
        if isinstance(module_args, str):
            try:
                module_args = eval(module_args)
            except Exception as e:
                print(e)
        model = get_model(task=task, model_name=model_name, model_args=model_args, data_module=data_module)
        model_kargs = dict(model_name=model_name, model=model, lr=lr,
                           output_dir=output_dir, model_args=model_args, data_args=data_args, module_args=module_args,
                           data_module=data_module)
        if task == 'rlsl':
            model_module = DiabetesRLSLModelModule(**model_kargs)
        elif task == 'sl':
            model_module = DiabetesSLModelModule(**model_kargs)
        elif task == 'patient':
            model_module = DiabetesPatientModelModule(**model_kargs)

        # trainer
        log_dir = output_dir / 'log'
        logger_csv = CSVLogger(str(log_dir))
        version_dir = Path(logger_csv.log_dir)

        trainer = Trainer(
            gpus=self._gpus,
            max_epochs=n_epoch,
            logger=[
                ConsoleLogger(),
                logger_csv,
            ],
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=patience),
                ModelCheckpoint(dirpath=(version_dir / 'checkpoint'), filename='{epoch}-{val_loss:.3f}',
                                monitor="val_loss", mode="min", save_last=True),
                TQDMProgressBar(refresh_rate=1),
            ],
            strategy='ddp',
        )
        trainer.fit(model_module, datamodule=data_module)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        model_module = model_module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, **model_kargs)

        # test
        dl_test = data_module.test_dataloader()
        test_eval = trainer.validate(model_module, dataloaders=dl_test) 
        if trainer.global_rank == 0:
            Path(output_dir / 'test_eval.json').write_text(json.dumps(test_eval, indent=2))
            dl_test.dataset.df.to_csv(output_dir / 'test_data.csv', index=False)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        trainer.test(model_module, dataloaders=dl_test)

        # merge test results
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if trainer.global_rank == 0:
            paths = sorted((output_dir / 'pred').glob('*.csv'))
            dfs_pred = []
            for path in paths:
                global_rank = int(path.name.split('.')[1])
                df_pred = pd.read_csv(path)
                df_pred['sample_id'] = df_pred['sample_id'] * trainer.world_size + global_rank
                dfs_pred += [df_pred]
            if len(dfs_pred) > 0:
                df_pred = pd.concat(dfs_pred).sort_values('sample_id')
                df_pred.to_csv(output_dir / 'test_pred.csv', index=False)

        if trainer.global_rank == 0:
            paths = sorted((output_dir / 'pred').glob('*.npy'))
            keys = list(set(path.name.split('.')[-3] for path in paths))
            for key in keys:
                npys = {}
                for path in paths:
                    global_rank = int(path.name.split('.')[-2])
                    npy = np.load(path)
                    k = path.name.split('.')[-3]
                    if k != key:
                        continue
                    npys[global_rank] = npy
                npys = [npys[k] for k in sorted(npys.keys())]
                npy = np.stack(npys, axis=1)
                npy = np.reshape(npy, [npy.shape[0] * npy.shape[1]] + list(npy.shape[2:]))

                output_test_pred = output_dir / 'test_pred'
                output_test_pred.mkdir(parents=True, exist_ok=True)
                np.save(str(output_test_pred / f'{key}.npy'), npy)

        # save model
        if trainer.global_rank == 0:
            model = model_module.model
            (output_dir / 'model_data.json').write_text(json.dumps(data_args, indent=2))
            torch.save(model.state_dict(), str(output_dir / 'state_dict.zip'))
            torch.save(model, str(output_dir / 'model.pt'))
            torch.save(model.state_dict(), str(version_dir / 'state_dict.zip'))
            torch.save(model, str(version_dir / 'model.pt'))

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def test(self, task, df_path, data_root, model_path, output_dir, col_group='dataset',
             data_args={}, model_args={}, module_args={}, n_step=7,
             ):
        """
        test

        Args:
        --------------------
        task: str
            Task Type(supports 4 types for now)
            - patient: patient model training or testing
            - rlsl: reinforcement learning with supervised learning
            - sl: supervised learning models
            - embedding: embedding representation
        df_path: str
            data table file path
            - contains at least two columns image_path, label
            - multiple regression tasks are stored in labels
        data_root: str
            dataset root path
        model_path: str
            model path
        output_dir: str
            result path
        """

        model_data_path = Path(model_path).parent / 'model_data.json'
        df_meta_path = Path(model_path).parent / 'task.columns.csv'
        output_dir = Path(output_dir)

        if model_data_path.exists():
            data_args_log = json.loads(model_data_path.read_text())
        else:
            data_args_log = {}
        print(data_args_log)

        # data module
        data_args = dict(data_args_log, **data_args)
        data_args['data_dir'] = data_args.pop('data_dir', data_root)
        data_args['df_meta_path'] = data_args.pop('df_meta_path', df_meta_path)
        data_module = DiabetesDataModule(df_path, task=task, data_args=data_args, col_group=col_group,
                                         batch_size=self._batch_size, num_workers=self._num_workers,
                                         pin_memory=self._pin_memory, shuffle=False,
                                         )
        data_module.setup()

        # model module
        model = get_model(task=task, model_path=model_path, data_module=data_module)
        model.n_step = n_step
        model_args['n_step'] = n_step
        model.df_meta_path = data_args.get('df_meta_path', df_meta_path)
        model_kargs = dict(model_name='test', model=model, output_dir=output_dir,
                           model_args=model_args, data_args=data_args, data_module=data_module, module_args=module_args)
        if task == 'rlsl':
            model_module = DiabetesRLSLModelModule(**model_kargs)
        if task == 'embedding':
            model_module = DiabetesRLSLTestEmbeddingModelModule(**model_kargs)
        elif task == 'sl':
            model_module = DiabetesSLModelModule(**model_kargs)
        elif task == 'patient':
            model_module = DiabetesPatientModelModule(**model_kargs)

        # trainer
        log_dir = output_dir / 'log'
        logger_csv = CSVLogger(str(log_dir))
        trainer = Trainer(
            gpus=self._gpus,
            max_epochs=0,
            logger=[
                logger_csv,
            ],
            callbacks=[
                TQDMProgressBar(refresh_rate=1),
            ],
            strategy='ddp',
        )

        # test
        data_module.setup()
        dl_test = data_module.test_dataloader()
        dl_test.dataset.df.to_csv(output_dir / 'test_data.csv', index=False)
        trainer.test(model_module, dataloaders=dl_test)

        # merge test results
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if trainer.global_rank == 0:
            paths = sorted((output_dir / 'pred').glob('*.npy'))
            keys = list(set(path.name.split('.')[-3] for path in paths))
            for key in keys:
                npys = {}
                for path in paths:
                    global_rank = int(path.name.split('.')[-2])
                    npy = np.load(path)
                    k = path.name.split('.')[-3]
                    if k != key:
                        continue
                    npys[global_rank] = npy
                npys = [npys[k] for k in sorted(npys.keys())]
                npy = np.stack(npys, axis=1)
                npy = np.reshape(npy, [npy.shape[0] * npy.shape[1]] + list(npy.shape[2:]))

                output_test_pred = output_dir / 'test_pred'
                output_test_pred.mkdir(parents=True, exist_ok=True)
                np.save(str(output_test_pred / f'{key}.npy'), npy)


if __name__ == '__main__':
    fire.Fire(DiabetesLightRunner)
