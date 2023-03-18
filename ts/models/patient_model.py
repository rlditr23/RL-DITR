
import torch
import torch.nn as nn
from typing import Optional, Any, Union, Callable

from ts.models.baseline import TransformerCausalEncoder, MLP, TransformerCausalDecoder


class PatientModel(nn.Module):
    '''
    def representation(self, observation):
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        return next_encoded_state_normalized, reward

    def prediction(self, encoded_state):
        return policy, value

    def initial_inference(self, observation):
        return value, reward, policy_logits, encoded_state,
    def recurrent_inference(self, encoded_state, action):
        return value, reward, policy_logits, next_encoded_state
    '''

    def __init__(self, n_reward_max, n_value_max, n_aux,
                 n_input, n_hidden=256, nhead=8, nhid=2048, nlayers=3, dropout: float = 0.25,
                 max_len=512, n_step=7,
                 ):
        super(PatientModel, self).__init__()

        self.model_type = 'Transformer'
        self.n_reward_max = n_reward_max
        self.n_value_max = n_value_max

        self.max_len = max_len
        self.n_step = n_step

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.nhead = nhead
        self.nlayers = nlayers
        self.nhid = nhid
        self.dropout = dropout
        self.n_reward_size = n_reward_max * 2 + 1
        self.n_value_size = n_value_max * 2 + 1

        self.encoder = TransformerCausalEncoder(n_input + n_hidden*2, n_hidden, nhead, nhid, nlayers, max_len=max_len, dropout=dropout)
        self.embedding_action = nn.Embedding(100 + 1, n_hidden)
        self.embedding_option = nn.Embedding(100 + 1, n_hidden)

        self.output_value_func = MLP(n_hidden, [n_hidden], self.n_value_size)
        self.output_reward_func = MLP(n_hidden*3, [n_hidden], self.n_reward_size)
        self.output_glu_func = MLP(n_hidden, [n_hidden], 1)
        self.output_aux_func = MLP(n_hidden, [n_hidden], n_aux)

        self.dynamics_func = TransformerCausalDecoder(n_hidden*3, n_hidden, nhead, nhid, nlayers, max_len=max_len, dropout=dropout)

    def ext_feat(self, state, action, option):
        action_embed = self.embedding_action(action)
        option_embed = self.embedding_option(option)
        feature = torch.cat([state, action_embed, option_embed], dim=-1)
        return feature

    @torch.jit.export
    def representation(self, obs, action_prev, option, padding_mask: Optional[torch.Tensor] = None, dynamics: bool = False):
        feature = self.ext_feat(obs, action_prev, option)
        state = self.encoder(feature, padding_mask, dynamics)
        return state

    @torch.jit.export
    def prediction(self, state: torch.Tensor):
        value = self.output_value_func(state)
        return value

    @torch.jit.export
    def dynamics(self, state, action, option, state_0, padding_mask: Optional[torch.Tensor] = None):
        '''
        state: (..., t, dim)
        action: (..., )
        option: (..., )
        historical information is required
        '''
        feature = self.ext_feat(state, action, option)
        next_state = self.dynamics_func(feature,state_0,padding_mask=padding_mask)
        reward = self.output_reward_func(feature)
        return next_state, reward

    @torch.jit.export
    def initial_inference(self, obs, action_prev, option, padding_mask: Optional[torch.Tensor] = None, dynamics: bool = False):
        state = self.representation(obs, action_prev, option, padding_mask, dynamics)
        value = self.prediction(state)
        return state, value

    @torch.jit.export
    def recurrent_inference(self, state, action, option, padding_mask: Optional[torch.Tensor] = None):
        '''
        state: (..., dim)
        action: (..., )
        option: (..., )
        '''
        next_state, reward = self.dynamics(state, action, option, padding_mask)
        value = self.prediction(next_state)
        return value, reward, next_state

    def forward(self, obs, action_prev, option, padding_mask: Optional[torch.Tensor] = None, n_step=None):
        '''
        obs: (B,L,E) float
        action: (B,L) long
        option: (B,L) long
        padding_mask: (B,L) (bool)
            - True Non zero means ignored
            - bool tensor
            - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

        Outputs:
        ---------------------
        state: (B,H,L,D)
        policy: (B,H,L,D)
        value: (B,H,L,D)
        reward: (B,H,L,D)
        aux: (B,H,L,D)
        glu: (B,H,L,D)

        H: lookahead
        '''
        if n_step is None:
            n_step = self.n_step

        state_0 = self.representation(obs, action_prev, option, padding_mask, dynamics=False)
        state_t = state_0
        action_t = torch.roll(action_prev, shifts=-1, dims=1)
        option_t = option

        state_list = []
        value_list = []
        reward_list = []
        glu_list = []
        aux_list = []
        for i in range(n_step): 
            state_t_next, reward_t = self.dynamics(state_t, action_t, option_t, padding_mask=padding_mask, state_0=state_0)
            value_t = self.prediction(state_t_next) 
            glu_t = self.output_glu_func(state_t_next)
            aux_t = self.output_aux_func(state_t_next)

            state_list += [state_t]
            value_list += [value_t]
            reward_list += [reward_t]
            glu_list += [glu_t]
            aux_list += [aux_t]

            state_t = state_t_next
            action_t = torch.roll(action_t, shifts=-1, dims=1)
            option_t = torch.roll(option_t, shifts=-1, dims=1)

        state = torch.stack(state_list, dim=1)
        value = torch.stack(value_list, dim=1)
        reward = torch.stack(reward_list, dim=1)
        glu = torch.stack(glu_list, dim=1)
        aux = torch.stack(aux_list, dim=1)

        outputs = {
            'state': state,
            'value': value,
            'reward': reward,
            'glu': glu,
            'aux': aux,
        }
        return outputs
