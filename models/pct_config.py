# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

pct_config = edict()

pct_config.NETWORK = edict()
pct_config.NETWORK.pose_dim = 15
pct_config.NETWORK.VOTE = True
pct_config.NETWORK.REFINE = True

pct_config.LOSS = edict()
pct_config.LOSS.VOTE_LOSS = True
pct_config.LOSS.PRIOR_LOSS = True


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in pct_config[k]:
            pct_config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in pct_config.py".format(k, vk))


def update_pct_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in pct_config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        pct_config[k][0] = (tuple(v))
                    else:
                        pct_config[k] = v
            else:
                raise ValueError("{} not exist in pct_config.py".format(k))


def gen_config(config_file):
    cfg = dict(pct_config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
