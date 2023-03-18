import numpy as np
import torch
from torch.nn import functional as F

from ts.pl_module.base import BaseModule
from ts.utils import logit_regression_loss, masked_loss, logit_regression_mae, support_to_scalar


class DiabetesPatientModelModule(BaseModule):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        self.loss_state_weight = self.module_args.get('loss_state_weight', 0.05)

    def forward(self, x):
        out = self.model(*x)
        return out

    def get_loss(self, pred_data, label_data):
        mask_reward = label_data['mask_reward']  # (b, l)
        aux = label_data['aux']
        glu_target = label_data['glu_target']
        mask_aux = label_data['mask_aux']
        reward = label_data['reward']
        cumreward = label_data['cumreward']
        padding = label_data['padding']

        pred_aux = pred_data['aux']
        pred_glu = pred_data['glu']
        pred_state = pred_data['state']
        pred_value = pred_data['value']
        pred_reward = pred_data['reward']

        b, n_step, length = reward.shape[:3]
        device = reward.device

        # mask
        step_mask = torch.ones([length], device=device)  # (batch, )
        step_masks = [step_mask]
        for i in range(n_step - 1):
            step_mask = torch.cat([torch.zeros_like(step_mask[-1:]), step_mask[:-1]])
            step_masks += [step_mask]
        step_mask = torch.stack(step_masks)  # [n_step, len]
        step_mask = step_mask[None, ...].expand(b, n_step, length)  # (b, n, l)
        step_mask = step_mask.bool()
        step_mask = step_mask

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
        loss_state = 0
        if n_step > 1:
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

        # total loss
        loss = loss_glu + loss_aux + loss_value + loss_reward + self.loss_state_weight * loss_state
        losses = {
            '_step_mask': step_mask,
            '_mask_reward': mask_reward,
            '_glu_mask': glu_mask,

            'loss': loss,
            'loss_value': loss_value,
            'loss_reward': loss_reward,
            'loss_aux': loss_aux,
            'loss_glu': loss_glu,
            'loss_state': loss_state,
        }
        return losses

    def training_step(self, batch, batch_idx):
        x, y = batch
        label_data = y
        pred_data = self.model(*x)
        label_data, stack_mask = self.label_data_stack(label_data, self.n_step)
        losses = self.get_loss(pred_data, label_data)
        loss = losses['loss']
        self.train_loss = loss.detach()
        return loss

    def validation_step(self, batch, batch_idx):
        x, label_data = batch
        pred_data = self.model(*x)
        label_data, stack_mask = self.label_data_stack(label_data, self.n_step)
        losses = self.get_loss(pred_data, label_data)
        loss = losses['loss']

        if self.train_loss is None:
            self.train_loss = 0
        self.log("train_loss", self.train_loss, prog_bar=False, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)

        glu_target = label_data['glu_target']
        reward = label_data['reward']
        cumreward = label_data['cumreward']
        pred_glu = pred_data['glu']
        pred_value = pred_data['value']
        pred_reward = pred_data['reward']

        # mask
        step_mask = losses['_step_mask']
        glu_mask = losses['_glu_mask']
        mask_reward = losses['_mask_reward']

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
            'loss_value': losses['loss_value'],
            'loss_reward': losses['loss_reward'],
            'loss_aux': losses['loss_aux'],
            'loss_glu': losses['loss_glu'],
            'loss_state': losses['loss_state'],
            'value_mae': value_mae,
            'reward_mae': reward_mae,
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
        pred_data = self.model(*x)
        obs_x, action_x, option_x, padding_x = x
        pred_value = pred_data['value']
        pred_reward = pred_data['reward']
        pred_aux = pred_data['aux']
        pred_glu = pred_data['glu']

        aux = label_data['aux']
        glu_target = label_data['glu_target']
        mask_aux = label_data['mask_aux']
        reward = label_data['reward']
        cumreward = label_data['cumreward']

        pred_aux = torch.sigmoid(pred_aux)
        pred_value = support_to_scalar(pred_value)
        pred_reward = support_to_scalar(pred_reward)

        result = {}
        result['padding_mask'] = padding_x.to('cpu')
        result['stack_mask'] = stack_mask.to('cpu')
        result['pred_value'] = pred_value.to('cpu')
        result['pred_reward'] = pred_reward.to('cpu')
        result['pred_aux'] = pred_aux.to('cpu')
        result['pred_glu'] = pred_glu.to('cpu')
        result['aux'] = aux.to('cpu')
        result['glu_target'] = glu_target.to('cpu')
        result['mask_aux'] = mask_aux.to('cpu')
        result['reward'] = reward.to('cpu')
        result['cumreward'] = cumreward.to('cpu')

        return result

    def test_epoch_end(self, outputs):
        global_rank = self.global_rank
        keys = list(outputs[0].keys())
        for key in keys:
            v = torch.cat([x[key] for x in outputs], dim=0)
            v = v.numpy()
            output_path = self.output_dir / f'test_pred.{key}.{global_rank}.npy'
            np.save(str(output_path), v)
