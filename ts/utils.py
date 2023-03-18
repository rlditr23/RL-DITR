import torch
from torch.nn import functional as F

def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1)
    logits = logits.to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits


def support_to_scalar(logits):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=-1)
    support_size = logits.shape[-1]//2
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=-1)
    return x


def logit_regression_loss(pred, true, mask=None):
    support_size = (pred.shape[-1] - 1) // 2
    if len(true.shape) == 3:
        b,n,l = true.shape
        true_dist = scalar_to_support(true.flatten(0,1), support_size)  # process positive and negative values
        true_dist = true_dist.reshape(*([b,n]+list(true_dist.shape[1:])))
    else:
        true_dist = scalar_to_support(true, support_size)  # process positive and negative values
    pred_logprob = F.log_softmax(pred, dim=-1)
    loss_all = -(true_dist * pred_logprob).mean(dim=-1)  # cross entropy (b, n, l)
    if mask is None:
        loss = loss_all.mean()
    else:
        loss = (loss_all * mask).sum() / mask.sum().clip(1)
    return loss


def logit_regression_mae(pred, true, mask=None):
    pred = support_to_scalar(pred)
    mae_all = torch.abs(true-pred)
    if mask is None:
        mae = mae_all.mean()
    else:
        mae = (mae_all*mask).sum()/mask.sum().clip(1)
    return mae


def masked_loss(loss_func, pred, true, mask=None):
    if mask is not None:
        if pred.shape > true.shape:
            true = true.expand_as(pred)
        else:
            pred = pred.expand_as(true)
        losses = loss_func(pred, true, reduction='none')
        loss = (losses * mask).sum() / mask.sum().clip(1)
    else:
        loss = loss_func(pred, true, reduction='mean')
    return loss


def masked_mean(values, mask=None):
    if mask is not None:
        mean = (values * mask).sum() / mask.sum().clip(1)
    else:
        mean = values.mean()
    return mean