import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

model_urls = {
    "0.25x": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/kinetics_shufflenetv2_0.25x_RGB_16_best.pth",
    "1.0x": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/kinetics_shufflenetv2_1.0x_RGB_16_best.pth",
    "1.5x": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/kinetics_shufflenetv2_1.5x_RGB_16_best.pth",
    "2.0x": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/kinetics_shufflenetv2_2.0x_RGB_16_best.pth",
}


def load_weight(model, arch):
    url = model_urls[arch]
    # check
    if url is None:
        print('No pretrained weight for 3D CNN: {}'.format(arch.upper()))
        return model

    # checkpoint state dict
    checkpoint = load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)

    checkpoint_state_dict = checkpoint.pop('state_dict')

    # model state dict
    model_state_dict = model.state_dict()
    # reformat checkpoint_state_dict:
    new_state_dict = {}
    for k in checkpoint_state_dict.keys():
        v = checkpoint_state_dict[k]
        new_state_dict[k[7:]] = v
    # pdb.set_trace()
    # check
    for k in list(new_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(new_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                new_state_dict.pop(k)
        else:
            new_state_dict.pop(k)

    model.load_state_dict(new_state_dict)

    return model


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=(5, 3, 3), stride=stride, padding=(2, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.stride == 1:
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, (5, 3, 3), stride, (2, 1, 1), groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True)
            )

        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv3d(inp, inp, (5, 3, 3), stride, (2, 1, 1), groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                # pw-linear
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True)
            )
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, (5, 3, 3), stride, (2, 1, 1), groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True)
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, depth, height, width)
    # permute
    x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x


class ShuffleNetV2(nn.Module):
    def __init__(self, width_mult='1.0x', num_classes=600):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == '0.25x':
            self.stage_out_channels = [-1, 24, 32, 64, 128]
        elif width_mult == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192]
        elif width_mult == '1.0x':
            self.stage_out_channels = [-1, 24, 128, 256, 512]
        elif width_mult == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704]
        elif width_mult == '2.0x':
            self.stage_out_channels = [-1, 24, 224, 488, 976]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, stride=(1, 2, 2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.features = []
        self.features1 = []
        self.features2 = []
        self.features3 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        # for idxstage in range(len(self.stage_repeats)):
        #     numrepeat = self.stage_repeats[idxstage]
        #     output_channel = self.stage_out_channels[idxstage+2]
        #     for i in range(numrepeat):
        #         if idxstage==0:
        #             stride = 2 if i == 0 else 1
        #             self.features1.append(InvertedResidual(input_channel, output_channel, stride))
        #             input_channel = output_channel
        #         elif idxstage==1:
        #             stride = 2 if i == 0 else 1
        #             self.features2.append(InvertedResidual(input_channel, output_channel, stride))
        #             input_channel = output_channel
        #         elif idxstage==2:
        #             stride = 2 if i == 0 else 1
        #             self.features3.append(InvertedResidual(input_channel, output_channel, stride))
        #             input_channel = output_channel
        # # make it nn.Sequential
        # self.features1 = nn.Sequential(*self.features1)
        # self.features2 = nn.Sequential(*self.features2)
        # self.features3 = nn.Sequential(*self.features3)

        # # building last several layers
        # self.conv_last      = conv_1x1x1_bn(input_channel, self.stage_out_channels[-1])
        # self.avgpool        = nn.AvgPool3d((2, 1, 1), stride=1)

    def forward(self, x):
        outputs = {}
        # pdb.set_trace()  #(1,3,16,512,512)     #(1,3,5,512,512)
        x = self.conv1(x)  # (1,24,16,256,256)    #(1,24,5,256,256)

        x = self.maxpool(x)  # (1,24,8,128,128)   #(1,24,3,128,128)
        # outputs['stage1'] = x
        # x=self.features(x)
        x = self.features[:4](x)  # (1,116,4,64,64) #(1,116,2,64,64)
        outputs['stage2'] = x  # torch.mean(x, dim=2, keepdim=True).squeeze(2)
        x = self.features[4:12](x)  # (1,232,2,32,32) #(1,232,1,32,32)
        outputs['stage3'] = x  # torch.mean(x, dim=2, keepdim=True).squeeze(2)
        x = self.features[12:16](x)  # (1,464,1,16,16) #(1,464,1,16,16)
        outputs['stage4'] = x  # torch.mean(x, dim=2, keepdim=True).squeeze(2)
        # out = self.conv_last(out)

        # if x.size(2) > 1:
        #     x = torch.mean(x, dim=2, keepdim=True)

        # return x.squeeze(2)
        return outputs


def build_shufflenetv2_3d(model_size='0.25x', pretrained=False):
    model = ShuffleNetV2(model_size)
    feats = model.stage_out_channels[-1]

    # if pretrained:
    #     model = load_weight(model, model_size)

    return model, feats


def build_3d_cnn(cfg, pretrained=False):
    if 'resnet' in cfg['backbone_3d']:
        model, feat_dims = build_resnet_3d(
            model_name=cfg['backbone_3d'],
            pretrained=pretrained
        )
    elif 'resnext' in cfg['backbone_3d']:
        model, feat_dims = build_resnext_3d(
            model_name=cfg['backbone_3d'],
            pretrained=pretrained
        )
    elif 'shufflenetv2' in cfg['backbone_3d']:
        model, feat_dims = build_shufflenetv2_3d(
            model_size=cfg['model_size'],
            pretrained=pretrained
        )
    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dims


class Backbone3D(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg

        # 3D CNN
        self.backbone, self.feat_dim = build_3d_cnn(cfg, pretrained)

    def forward(self, x):
        """
            Input:
                x: (Tensor) -> [B, C, T, H, W]
            Output:
                y: (List) -> [
                    (Tensor) -> [B, C1, H1, W1],
                    (Tensor) -> [B, C2, H2, W2],
                    (Tensor) -> [B, C3, H3, W3]
                    ]
        """
        feat = self.backbone(x)

        return feat

