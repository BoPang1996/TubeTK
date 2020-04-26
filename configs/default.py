import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
INF = 1e8

__C = edict()
cfg = __C

# for generating the tubes
__C.min_visibility = -0.1
__C.tube_thre = 0.8
__C.forward_frames = 4
__C.frame_stride = 1
__C.value_range = 1
__C.img_size = [896, 1152]

# pretrain
__C.pretrain_model_path = ''

# for ResNet
__C.freeze_stages = -1
__C.backbone = 'res50'

# for FPN
__C.fpn_features_n = 256
__C.fpn_outs_n = 5

# for FCOS head
__C.tube_points = 14
__C.heads_features_n = 256
__C.heads_layers_n = 4
__C.withoutThickCenterness = False
__C.model_stride = [[2, 8],
                    [4, 16],
                    [8, 32],
                    [8, 64],
                    [8, 128]]
__C.regress_range = ([(-1, 0.25), (-1, 0.0714)],
                     [(0.25, 0.5), (0.0714, 0.1428)],
                     [(0.5, 0.75), (0.1428, 0.2857)],
                     [(0.75, INF), (0.2857, 0.5714)],
                     [(0.75, INF), (0.5714, INF)])


# for loss
__C.reg_loss = 'giou'
__C.tube_limit = 1000
__C.test_nms_pre = 1000
__C.test_nms_max_per_img = 500
__C.test_nms_score_thre = 0.5
__C.test_nms_iou_thre = 0.5
__C.linking_min_iou = 0.4
__C.cos_weight = 0.2


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
