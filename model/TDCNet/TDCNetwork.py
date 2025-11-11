import torch
import torch.nn as nn

from model.TDCNet.TDCSTA import CrossAttention, SelfAttention
from model.TDCNet.backbone3d import Backbone3D
from model.TDCNet.backbonetd import BackboneTD
from model.TDCNet.darknet import BaseConv, CSPDarknet, DWConv


class Feature_Backbone(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features

    def forward(self, input):
        out_features = self.backbone.forward(input)
        [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]
        return [feat1, feat2, feat3]


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = BaseConv  # if depthwise else BaseConv
        # --------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        # --------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        # self.conv2=nn.Identity()
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class FusionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5, depthwise=False, act="silu", ):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        n = 1
        # --------------------------------------------------#
        #   主干部分的初次卷积
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   大的残差边部分的初次卷积
        # --------------------------------------------------#
        self.conv2 = BaseConv(hidden_channels, hidden_channels, 1, stride=1, act=act)  # in_channel
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # self.deepfeature=nn.Sequential(BaseConv(hidden_channels, hidden_channels//2, 1, stride=1, act=act),
        #       BaseConv(hidden_channels//2, hidden_channels, 3, stride=1, act=act))
        # -----------------------------------------------#
        # module_list = [Bottleneck(hidden_channels, hidden_channels, True, 1.0, depthwise, act=act) for _ in range(n)]
        # self.deepfeature      = nn.Sequential(*module_list)
        self.conv3 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)  # 2*hidden_channel

        # --------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        # --------------------------------------------------#
        # module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        # self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        # -------------------------------#
        #   x_1是主干部分
        # -------------------------------#
        # x_1 = self.conv1(x)
        x = self.conv1(x)
        # -------------------------------#
        #   x_2是大的残差边部分
        # -------------------------------#
        # x_2 = self.conv2(x)
        x = self.conv2(x)
        # -----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        # -----------------------------------------------#
        # x_1 = self.deepfeature(x_1)
        # -----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        # -----------------------------------------------#
        # x = torch.cat((x_1, x_2), dim=1)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        return self.conv3(x)


class Feature_Fusion(nn.Module):
    def __init__(self, in_channels=[128, 256, 512], depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        # self.lateral_conv0 = BaseConv(2 * int(in_channels[2]), int(in_channels[1]), 1, 1, act=act)
        self.lateral_conv0 = BaseConv(in_channels[1] + in_channels[2], in_channels[1], 1, 1, act=act)

        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        self.C3_p4 = FusionLayer(
            int(2 * in_channels[1]),
            int(in_channels[1]),
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        # self.reduce_conv1 = BaseConv(int(2 * in_channels[1]), int(in_channels[0]), 1, 1, act=act)
        self.reduce_conv1 = BaseConv(int(in_channels[0] + in_channels[1]), int(in_channels[0]), 1, 1, act=act)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        self.C3_p3 = FusionLayer(
            int(2 * in_channels[0]),
            int(in_channels[0]),
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        out_features = input  # self.backbone.forward(input)
        [feat1, feat2, feat3] = out_features  # [out_features[f] for f in self.in_features]

        # -------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        # -------------------------------------------#
        # P5          = self.lateral_conv0(feat3)
        # -------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        # -------------------------------------------#
        P5_upsample = self.upsample(feat3)
        # -------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        # -------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        # pdb.set_trace()
        # -------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        # -------------------------------------------#
        P4 = self.lateral_conv0(P5_upsample)
        # P5_upsample = self.C3_p4(P5_upsample)

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        # P4          = self.reduce_conv1(P5_upsample)
        # -------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        # -------------------------------------------#
        P4_upsample = self.upsample(P4)
        # -------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        # -------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        P3_out = self.reduce_conv1(P4_upsample)
        # P3_out      = self.C3_p3(P4_upsample)

        return P3_out


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[16, 32, 64], act="silu"):
        super().__init__()
        Conv = BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int(in_channels[i]), out_channels=int(256 * width), ksize=1, stride=1, act=act))  # 128-> 256 通道整合
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            # ---------------------------------------------------#
            x = self.stems[k](x)
            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            cls_feat = self.cls_convs[k](x)
            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)

            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)
            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


model_config = {

    'backbone_2d': 'yolo_free_nano',
    'pretrained_2d': True,
    'stride': [8, 16, 32],
    # ## 3D
    'backbone_3d': 'shufflenetv2',
    'model_size': '1.0x',  # 1.0x
    'pretrained_3d': True,
    'memory_momentum': 0.9,
    'head_dim': 128,  # 64
    'head_norm': 'BN',
    'head_act': 'lrelu',
    'num_cls_heads': 2,
    'num_reg_heads': 2,
    'head_depthwise': True,

}


def build_backbone_3d(cfg, pretrained=False):
    backbone = Backbone3D(cfg, pretrained)
    return backbone, backbone.feat_dim


mcfg = model_config


class TDCNetwork(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=5):
        super(TDCNetwork, self).__init__()
        self.num_frame = num_frame
        self.backbone2d = Feature_Backbone(0.33, 0.50)
        self.backbone3d, bk_dim_3d = build_backbone_3d(mcfg, pretrained=mcfg['pretrained_3d'] and True)
        self.backbonetd = BackboneTD(mcfg, pretrained=mcfg['pretrained_3d'] and True)
        self.q_sa1 = SelfAttention(128, window_size=(2, 8, 8), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.k_sa1 = SelfAttention(128, window_size=(2, 8, 8), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.v_sa1 = SelfAttention(128, window_size=(2, 8, 8), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.q_sa2 = SelfAttention(256, window_size=(2, 4, 4), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.k_sa2 = SelfAttention(256, window_size=(2, 4, 4), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.v_sa2 = SelfAttention(256, window_size=(2, 4, 4), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.q_sa3 = SelfAttention(512, window_size=(2, 2, 2), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.k_sa3 = SelfAttention(512, window_size=(2, 2, 2), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.v_sa3 = SelfAttention(512, window_size=(2, 2, 2), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.ca1 = CrossAttention(128, window_size=(2, 8, 8), num_heads=4)
        self.ca2 = CrossAttention(256, window_size=(2, 4, 4), num_heads=4)
        self.ca3 = CrossAttention(512, window_size=(2, 2, 2), num_heads=4)
        self.feature_fusion = Feature_Fusion()
        self.head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[128], act="silu")

    def forward(self, inputs):
        # inputs: [B, 3, T, H, W]
        if len(inputs.shape) == 5:
            T = inputs.shape[2]
            diff_imgs = inputs[:, :, :T // 2, :, :]
            mt_imgs = inputs[:, :, T // 2:, :, :]
        else:
            diff_imgs = inputs
            mt_imgs = inputs
        q_3d = self.backbonetd(diff_imgs)
        q_3d1, q_3d2, q_3d3 = q_3d['stage2'], q_3d['stage3'], q_3d['stage4']
        k_3d = self.backbone3d(mt_imgs)
        k_3d1, k_3d2, k_3d3 = k_3d['stage2'], k_3d['stage3'], k_3d['stage4']
        [feat1, feat2, feat3] = self.backbone2d(inputs[:, :, -1, :, :])

        def to_5d(x):
            # [B, C, T, H, W] -> [B, T, H, W, C]
            return x.permute(0, 2, 3, 4, 1)

        q_3d1 = to_5d(q_3d1)
        q_3d2 = to_5d(q_3d2)
        q_3d3 = to_5d(q_3d3)
        k_3d1 = to_5d(k_3d1)
        k_3d2 = to_5d(k_3d2)
        k_3d3 = to_5d(k_3d3)

        # V特征扩展T维度，与Q/K对齐（假设V为最后一帧，T=1）
        def expand_v(x, T):
            # [B, C, H, W] -> [B, T, H, W, C]，复制T次
            x = x.permute(0, 2, 3, 1).unsqueeze(1)
            x = x.expand(-1, T, -1, -1, -1)
            return x

        T1 = q_3d1.shape[1]
        T2 = q_3d2.shape[1]
        T3 = q_3d3.shape[1]
        v1 = expand_v(feat1, T1)
        v2 = expand_v(feat2, T2)
        v3 = expand_v(feat3, T3)

        q1 = self.q_sa1(q_3d1)
        k1 = self.k_sa1(k_3d1)
        v1 = self.v_sa1(v1)
        q2 = self.q_sa2(q_3d2)
        k2 = self.k_sa2(k_3d2)
        v2 = self.v_sa2(v2)
        q3 = self.q_sa3(q_3d3)
        k3 = self.k_sa3(k_3d3)
        v3 = self.v_sa3(v3)
        out1 = self.ca1(q1, k1, v1)
        out2 = self.ca2(q2, k2, v2)
        out3 = self.ca3(q3, k3, v3)
        out1 = out1.mean(1).permute(0, 3, 1, 2)
        out2 = out2.mean(1).permute(0, 3, 1, 2)
        out3 = out3.mean(1).permute(0, 3, 1, 2)

        feat_all = self.feature_fusion([out1, out2, out3])
        outputs = self.head([feat_all])

        return outputs