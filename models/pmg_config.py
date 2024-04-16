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

config = edict()

config.NETWORK = edict()
config.NETWORK.USE_PCD = True
config.NETWORK.UP_TYPE = 'graph_mlp' #'upsample'
config.NETWORK.ADJ_TYPE = 'adj' #'adj', laplacian
config.NETWORK.PCD_LAST = False #'adj'
config.NETWORK.PRN_TRAINED = True
config.NETWORK.PRN_PRETRAINED = True

config.LOSS = edict()
config.LOSS.MID_LOSS = True
config.LOSS.MID_EDGE_LOSS = False

def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
