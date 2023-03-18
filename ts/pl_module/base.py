from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn


class BaseModule(pl.LightningModule):
    def __init__(self, model_name, model, output_dir=None,
                 lr=0.001, model_args=None, data_args=None, module_args=None, data_module=None,
                 ):
        super().__init__()
        if model_args is None:
            model_args = {}
        if data_args is None:
            data_args = {}
        self.lr = lr
        self.model_name = model_name
        self.model_args = model_args
        self.model = model
        self.data_args = data_args
        self.module_args = module_args
        self.data_module = data_module

        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # metic
        self.train_loss = None
        metrics = {
            'loss_aux': torchmetrics.MeanMetric(),
            'loss_glu': torchmetrics.MeanMetric(),
            'loss_action': torchmetrics.MeanMetric(),

            'loss_value': torchmetrics.MeanMetric(),
            'loss_reward': torchmetrics.MeanMetric(),
            'loss_state': torchmetrics.MeanMetric(),

            'loss_rl': torchmetrics.MeanMetric(),

            'action_mae': torchmetrics.MeanMetric(),
            'glu_mae': torchmetrics.MeanMetric(),

            'value_mae': torchmetrics.MeanMetric(),
            'reward_mae': torchmetrics.MeanMetric(),
        }
        self.metrics = nn.ModuleDict(metrics)

        self.n_step = self.model_args.get('n_step', 1)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_start(self):
        log_hyperparams = {
            "model_name": self.model_name,
            "data_args": self.data_args,
            "model_args": self.model_args,
            "module_args": self.module_args,
            "lr": self.lr,
        }
        log_hyperparams.update(self.model_args)
        self.logger.log_hyperparams(log_hyperparams)

    def label_data_stack(self, label_data, n_step):
        keys = [
            'action',
            'mask_reward',
            'mask_aux',
            'aux',
            'glu_target',
            'mask_aux',
            'reward',
            'cumreward',
        ]

        # create mask
        mask = torch.ones_like(label_data['glu_target'])
        masks = []
        for i in range(n_step):
            mask_t = torch.roll(mask, shifts=-i, dims=1)
            mask_t[:, -1] = 0
            masks += [mask_t]
            mask = mask_t
        stack_mask = torch.stack(masks, dim=1)

        # create stack label
        stack_label = {}
        for key in keys:
            val = label_data[key]
            vals = []
            for i in range(n_step):
                val_t = torch.roll(val, shifts=-i, dims=1)
                vals += [val_t]
            val = torch.stack(vals, dim=1)
            stack_label[key] = val

        stack_label['padding'] = label_data['padding']

        return stack_label, stack_mask

    def test_epoch_end(self, outputs):
        global_rank = self.global_rank
        keys = list(outputs[0].keys())
        for key in keys:
            v = torch.cat([x[key] for x in outputs], dim=0)
            v = v.numpy()
            output_path = self.output_dir / f'test_pred.{key}.{global_rank}.npy'
            np.save(str(output_path), v)
