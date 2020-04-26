import time, os
import torch
import torch.nn as nn
from network.resnet import resnet101, resnet50, resnext101
from network.fpn import FPN
from network.track_head import TrackHead


class TubeTK(nn.Module):

    def __init__(self,
                 num_classes,
                 arg,
                 pretrained=True
                 ):
        super(TubeTK, self).__init__()
        self.arg = arg
        if arg.backbone == 'res50':
            self.backbone = resnet50(freeze_stages=arg.freeze_stages, fst_l_stride=arg.model_stride[0][0])
        elif arg.backbone == 'res101':
            self.backbone = resnet101(freeze_stages=arg.freeze_stages, fst_l_stride=arg.model_stride[0][0])
        elif arg.backbone == 'resx101':
            self.backbone = resnext101(freeze_stages=arg.freeze_stages, fst_l_stride=arg.model_stride[0][0])
        else:
            raise NotImplementedError
        self.neck = FPN(in_channels=[512, 1024, 2048], arg=arg)
        self.tube_head = TrackHead(arg=arg,
                                   num_classes=num_classes,
                                   in_channels=self.neck.out_channels,
                                   strides=[[arg.model_stride[i][0]/(arg.forward_frames * 2) * arg.value_range,
                                            arg.model_stride[i][1]/arg.img_size[0] * arg.value_range,
                                            arg.model_stride[i][1]/arg.img_size[1] * arg.value_range] for i in range(5)]
                                   )

        if pretrained and arg.pretrain_model_path != '':
            self.load_pretrain(model_path=arg.pretrain_model_path)
        torch.cuda.empty_cache()

    def load_pretrain(self, model_path):
        if int(os.environ["RANK"]) == 0:
            print('loading JTA Pretrain: ' + str(model_path))

        pre_model = torch.load(model_path, map_location={'cuda:0': 'cpu',
                                                         'cuda:1': 'cpu',
                                                         'cuda:2': 'cpu',
                                                         'cuda:3': 'cpu',
                                                         'cuda:4': 'cpu',
                                                         'cuda:5': 'cpu',
                                                         'cuda:6': 'cpu',
                                                         'cuda:7': 'cpu'})['state']
        model_dict = self.state_dict()
        for key in model_dict:
            if model_dict[key].shape != pre_model['module.' + key].shape:
                p_shape = model_dict[key].shape
                pre_model['module.' + key] = pre_model['module.' + key].repeat(1, 1, p_shape[2], 1, 1) / p_shape[2]
            else:
                model_dict[key] = pre_model['module.' + key]
        self.load_state_dict(model_dict)
        del pre_model, model_dict

    def extract_feat(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_tubes,
                      gt_labels):
        x = self.extract_feat(img)
        outs = self.tube_head(x)
        loss_inputs = outs + (gt_tubes, gt_labels, img_metas)
        losses = self.tube_head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, img_meta):
        x = self.extract_feat(img)
        outs = self.tube_head(x)
        tube_inputs = outs + (img_meta, self.arg)
        tube_list = self.tube_head.get_tubes(*tube_inputs)
        return tube_list

    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta)
