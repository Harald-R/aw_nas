# -*- coding: utf-8 -*-
#pylint: disable=attribute-defined-outside-init

import os
import collections

import numpy as np
import scipy
import scipy.signal

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class attr_dict_wrapper(object): #pylint: disable=invalid-name
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]

def compute_returns(rewards, gamma, length=None):
    if not isinstance(rewards, collections.Sequence):
        assert length is not None
        _rewards = np.zeros((length,))
        _rewards[-1] = rewards
    else:
        _rewards = rewards
    return scipy.signal.lfilter([1], [1, -gamma], _rewards[::-1], axis=0)[::-1]

def get_cfg_wrapper(cfg, key):
    if isinstance(cfg, dict):
        return cfg
    return {key: cfg}

def _assert_keys(dct, mandatory_keys, possible_keys, name):
    if mandatory_keys:
        assert set(mandatory_keys).issubset(dct.keys()),\
            "{} schedule cfg must have keys: ({})".format(name, ", ".join(mandatory_keys))
    if possible_keys:
        addi_keys = set(dct.keys()).difference(possible_keys)
        assert not addi_keys,\
            "{} schedule cfg cannot have keys: ({}); all possible keys: ({})"\
                .format(name, ", ".join(addi_keys), ", ".join(possible_keys))

_SUPPORTED_TYPES = {"value", "mul", "add"}
def check_schedule_cfg(schedule):
    """
    Check the sanity of the schedule configuration.
    Currently supported type: mul, add, value.

    Rules: mul  : [boundary / every], step, start, [optional: min, max]
           add  : [boundary / every], step, start, [optional: min, max]
           value: boundary, value
    """
    assert "type" in schedule,\
        "Schedule config must have `type` specified: one in "+", ".join(_SUPPORTED_TYPES)
    type_ = schedule["type"]
    assert type_ in _SUPPORTED_TYPES, "Supported schedule config type: "+", ".join(_SUPPORTED_TYPES)

    if type_ == "value":
        _assert_keys(schedule, ["value", "boundary"], None, "value")
        assert len(schedule["value"]) == len(schedule["boundary"]),\
            "value schedule cfg `value` and `boundary` should be of the same length."
        assert schedule["boundary"][0] == 1,\
            "value schedule cfg must have `boundary` config start from 1."
    else: # mul/add
        _assert_keys(schedule, ["step", "start"],
                     ["type", "step", "start", "boundary", "every", "min", "max"], "value")
        assert "boundary" in schedule or "every" in schedule,\
            "{} schedule cfg must have one of `boundary` and `every` key existed.".format(type_)
        assert not ("boundary" in schedule and "every" in schedule),\
            "{} shcedule cfg cannot have `boundary` and `every` key in the mean time.".format(type_)

def get_schedule_value(schedule, epoch):
    """
    See docstring of `check_schedule_cfg` for details.
    """

    type_ = schedule["type"]
    if type_ == "value":
        ind = list(np.where(epoch < np.array(schedule["boundary"]))[0])
        if not ind: # if epoch is larger than the last boundary
            ind = len(schedule["boundary"]) - 1
        else:
            ind = ind[0] - 1
        next_v = schedule["value"][ind]
    else:
        min_ = schedule.get("min", -np.inf)
        max_ = schedule.get("max", np.inf)
        if "every" in schedule:
            ind = (epoch - 1) // schedule["every"]
        else: # "boundary" in schedule
            ind = list(np.where(epoch < np.array(schedule["boundary"]))[0])
            if not ind: # if epoch is larger than the last boundary
                ind = len(schedule["boundary"])
            else:
                ind = ind[0]
        if type_ == "mul":
            next_v = schedule["start"] * schedule["step"] ** ind
        else: # type_ == "add"
            next_v = schedule["start"] + schedule["step"] * ind
        next_v = max(min(next_v, max_), min_)
    return next_v

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


#pylint: disable=invalid-name
if __name__ == "__main__":
    EPS = 1e-8
    cfg = {
        "type": "mul",
        "boundary": [6, 11],
        "step": 0.1,
        "start": 1.
    }
    assert get_schedule_value(cfg, 1) - 1 < EPS
    assert get_schedule_value(cfg, 5) - 1 < EPS
    assert get_schedule_value(cfg, 6) - 0.1 < EPS
    assert get_schedule_value(cfg, 12) - 0.01 < EPS

    cfg = {
        "type": "add",
        "every": 2,
        "step": -0.3,
        "start": 1.,
        "min": 0
    }
    assert get_schedule_value(cfg, 1) - 1.0 < EPS
    assert get_schedule_value(cfg, 2) - 1.0 < EPS
    assert get_schedule_value(cfg, 3) - 0.7 < EPS
    assert get_schedule_value(cfg, 5) - 0.4 < EPS
    assert get_schedule_value(cfg, 9) - 0. < EPS

    cfg = {
        "type": "value",
        "value": [None, 0.1, 1.0],
        "boundary": [1, 11, 21]
    }
    assert get_schedule_value(cfg, 2) is None
    assert get_schedule_value(cfg, 11) - 0.1 < EPS
    assert get_schedule_value(cfg, 40) - 1.0 < EPS
#pylint: enable=invalid-name
