import torch
from torch.nn import functional as F

from ts.pl_module.base import BaseModule
from ts.utils import support_to_scalar


class DiabetesRLSLTestEmbeddingModelModule(BaseModule):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def forward(self, x):
        out = self.model(*x)
        return out

    def training_step(self, batch, batch_idx):
        loss = 0
        self.train_loss = loss
        return loss

    def validation_step(self, batch, batch_idx):
        loss = 0
        return loss

    def test_step(self, batch, batch_idx):
        x, label_data = batch
        n_step = self.n_step
        label_data, stack_mask = self.label_data_stack(label_data, n_step=n_step)
        pred_data = self.model(*x)
        obs_x, action_x, option_x, padding_x = x
        pred_state = pred_data['state']
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
        result['pred_state'] = pred_state.to('cpu')
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