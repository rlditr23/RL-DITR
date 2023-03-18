import numpy as np
import pandas as pd


def glu2risk(glu):
    glu = glu * 18
    c0, c1, c2 = 1.509, 1.084, 5.381
    risk = 10 * (c0 * (np.log(glu) ** c1 - c2)) ** 2
    risk = 1 - np.clip(risk, a_min=0, a_max=7.75 * 2) / 7.75
    if isinstance(risk, np.ndarray) or isinstance(risk, pd.Series):
        risk[glu < 70] = -1
    else:
        if glu < 70:
            risk = -1
    return risk


def reward2return(rewards, gamma=0.9):
    rewards = rewards.fillna(0)
    result = np.zeros_like(rewards, dtype='float')
    steps = len(rewards)
    running_add = 0
    for i in reversed(range(steps)):
        running_add = running_add * gamma + rewards.values[i]
        result[i] = running_add
    result = pd.Series(result, index=rewards.index)
    return result


def glu2reward(glus, dtype='risk'):
    if dtype == 'risk':
        rewards = glu2risk(glus.clip(0.01))
    else:
        rewards = glus.between(3.9, 10) * 2 - 1
    rewards = rewards.clip(-1, 1)
    rewards[glus <= 0.01] = 0
    return rewards
