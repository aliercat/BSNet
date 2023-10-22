#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


from nets.stem_sesp_stdcnet_trt import STDCNet1446, STDCNet813

# from modules.bn import InPlaceABNSync as BatchNorm2d
BatchNorm2d = nn.BatchNorm2d

## dd
import torch.utils.model_zoo as modelzoo
from torch.nn import init
import math

backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'


## dd bisev2
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):  # 输入通道数，输出通道数，卷积核大小，步长，填充，扩张，这里是same
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


## dd
class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor  # 4倍？
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)  # 非插值上采样
        self.init_weight()

    def forward(self, x):
        #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


## dd
class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):  # 训练时用，因为需要边缘检测
        feat2 = self.S1(x)
        feat4 = self.S2(feat2)
        feat8 = self.S3(feat4)
        return feat2, feat4, feat8

    def forward_impl(self, x):  # 推理时用，不用边缘检测
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


# #为stdc的
# class ConvBNReLU(nn.Module):
#     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
#         super(ConvBNReLU, self).__init__()
#         self.conv = nn.Conv2d(in_chan,
#                 out_chan,
#                 kernel_size = ks,
#                 stride = stride,
#                 padding = padding,
#                 bias = False)
#         self.bn = BatchNorm2d(out_chan)
#         # self.bn = BatchNorm2d(out_chan, activation='none')
#         self.relu = nn.ReLU()
#         self.init_weight()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

# #DetailHead
# class BiSeNetOutput(nn.Module):
#     def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
#         super(BiSeNetOutput, self).__init__()
#         self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
#         self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
#         self.init_weight()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.conv_out(x)
#         return x

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#     def get_params(self):
#         wd_params, nowd_params = [], []
#         for name, module in self.named_modules():
#             if isinstance(module, (nn.Linear, nn.Conv2d)):
#                 wd_params.append(module.weight)
#                 if not module.bias is None:
#                     nowd_params.append(module.bias)
#             elif isinstance(module, BatchNorm2d):
#                 nowd_params += list(module.parameters())
#         return wd_params, nowd_params

# 语义分支
class ContextPath(nn.Module):
    def __init__(self, backbone='CatNetSmall', pretrain_model='', use_conv_last=False, *args, **kwargs):
        super(ContextPath, self).__init__()

        self.backbone_name = backbone
        if backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)

        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.init_weight()

    def forward(self, x):
        # 经过stage5的结果，因为没有其他arm操作了
        feat = self.backbone(x)
        # print()
        # print(feat.shape)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


# # New_BGA模块
# class BGALayer(nn.Module):
#     # C1=128,C2=1024
#     def __init__(self):
#         super(BGALayer, self).__init__()
#         # self.left2 = nn.Sequential(
#         #     nn.Conv2d(
#         #         128, 1024, kernel_size=3, stride=2,
#         #         padding=1, bias=False),
#         #     nn.BatchNorm2d(1024),
#         #     nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
#         # )
#         self.right1 = nn.Sequential(
#             nn.Conv2d(
#                 1024, 128, kernel_size=3, stride=1,
#                 padding=1, bias=False),
#             nn.BatchNorm2d(128),
#         )
#         self.up1 = nn.Upsample(scale_factor=4)
#         #self.up1 = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
#         # self.up2 = nn.Upsample(scale_factor=4)
#         # self.conv1 = nn.Sequential(
#         #         nn.Conv2d(
#         #             1024, 128, kernel_size=1, stride=1,
#         #             padding=0, bias=False),
#         #         nn.BatchNorm2d(128),
#         #         )

#         ##TODO: does this really has no relu?
#         # 加和后的处理
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 128, 128, kernel_size=3, stride=1,
#                 padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),  # not shown in paper
#         )

#     def forward(self, x_d, x_s):
#         dsize = x_s.size()[2:]  # dsize?
#         # print(x_s.shape)
#         # left2 = self.left2(x_d)
#         right1 = self.right1(x_s)

#         right1 = self.up1(right1)

#         left = x_d * torch.sigmoid(x_d)
#         left = left * (1 - torch.sigmoid(right1))

#         right = right1 * torch.sigmoid(right1)

#         right = right * (1 - torch.sigmoid(x_d))

#         # right = self.conv1(self.up2(right1)

#         # print("right最后\n")
#         # print(right.shape)
#         # 目前是变成同一通道后按照bisenet那样加和后再3*3relu
#         out = self.conv2(left + right)
#         return out

# # 把原来的FFM模块，换成了BGA模块
# class BGALayer(nn.Module):
# # C1=128,C2=1024
#     def __init__(self):
#         super(BGALayer, self).__init__()
#         # self.left2 = nn.Sequential(
#         #     nn.Conv2d(
#         #         128, 1024, kernel_size=3, stride=2,
#         #         padding=1, bias=False),
#         #     nn.BatchNorm2d(1024),
#         #     nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
#         # )
#         # self.right1 = nn.Sequential(
#         #     nn.Conv2d(
#         #         1024, 128, kernel_size=3, stride=1,
#         #         padding=1, bias=False),
#         #     nn.BatchNorm2d(128),
#         # )
#         # self.up1 = nn.Upsample(scale_factor=4)
#         self.up2 = nn.Upsample(scale_factor=4)
#         self.conv1 = nn.Sequential(
#                 nn.Conv2d(
#                     1024, 128, kernel_size=1, stride=1,
#                     padding=0, bias=False),
#                 nn.BatchNorm2d(128),
#                 )

#         ##TODO: does this really has no relu?
#         # 加和后的处理
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 128, 128, kernel_size=3, stride=1,
#                 padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True), # not shown in paper
#         )

#     def forward(self, x_d, x_s):
#         dsize = x_s.size()[2:]#dsize?
#         #print(x_s.shape)
#         # left2 = self.left2(x_d)
#         # right1 = self.right1(x_s)

#         # right1 = self.up1(right1)

#         left = x_d * torch.sigmoid(x_d)
#         #left = left * (1 - torch.sigmoid(right1))

#         right = x_s * torch.sigmoid(x_s)

#         # right = right *(1 - torch.sigmoid(left2))

#         right = self.conv1(self.up2(right))

#         # print("right最后\n")
#         # print(right.shape)
#         # 目前是变成同一通道后按照bisenet那样加和后再3*3relu
#         out = self.conv2(left + right)
#         return out

# 1-a添加进BGA模块
class BGALayer(nn.Module):
# C1=128,C2=1024
    def __init__(self):
        super(BGALayer, self).__init__()
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 1024, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                1024, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    1024, 128, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(128),
                )

        ##TODO: does this really has no relu?
        # 加和后的处理
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_s.size()[2:]#dsize?
        #print(x_s.shape)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)

        right1 = self.up1(right1)

        left = x_d * torch.sigmoid(x_d)
        left = left * (1 - torch.sigmoid(right1))

        right = x_s * torch.sigmoid(x_s)

        right = right *(1 - torch.sigmoid(left2))

        right = self.conv1(self.up2(right))

        # print("right最后\n")
        # print(right.shape)
        # 目前是变成同一通道后按照bisenet那样加和后再3*3relu
        out = self.conv2(left + right)
        return out

## bisev2的，没动
class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
            ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        # print()
        # print("feat.shape")
        # print(feat.shape)
        return feat


# 整个网络了，涉及边缘检测
class BiSeNet(nn.Module):
    def __init__(self, backbone, n_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False,
                 use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super(BiSeNet, self).__init__()

        # 对应的是detail branch的三阶段输出通道数，方便后续的边缘检测
        sp2_inplanes = 64
        sp4_inplanes = 64
        sp8_inplanes = 128

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        # self.heat_map = heat_map

        # 空间路径的结果
        self.detail = DetailBranch()

        # 语义分支的结果，但我没有预训练模型
        self.cp = ContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)

        # 融合结果
        self.bga = BGALayer()

        # 分割头，这里没有辅助所以aux=False
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)

        # 待定，原本用于 边缘检测，这里用不着了
        # if backbone == 'STDCNet1446':
        #     conv_out_inplanes = 128
        #     #原本用于 边缘检测，这里用不着了
        #     # sp2_inplanes = 32
        #     # sp4_inplanes = 64
        #     # sp8_inplanes = 256
        #     #sp16_inplanes = 512
        #     #inplane = sp8_inplanes + conv_out_inplanes

        # #待定

        # elif backbone == 'STDCNet813':
        #     conv_out_inplanes = 128
        #     # sp2_inplanes = 32
        #     # sp4_inplanes = 64
        #     # sp8_inplanes = 256
        #     #sp16_inplanes = 512
        #     #inplane = sp8_inplanes + conv_out_inplanes

        # else:
        #     print("backbone is not in backbone lists")
        #     exit(0)

        # detail head BiSeNetOutput 换成SegmentHead了
        self.conv_out_sp8 = SegmentHead(sp8_inplanes, 64, 1)
        # self.conv_out_sp4 = SegmentHead(sp4_inplanes, 64, 1)
        # self.conv_out_sp2 = SegmentHead(sp2_inplanes, 64, 1)
        self.init_weight()

    # dd
    def forward(self, x):
        H, W = x.size()[2:]

        # 空间分支
        feat_res2, feat_res4, feat_res8 = self.detail(x)
        # detail head
        # feat_out_sp2 = self.conv_out_sp2(feat_res2)
        # feat_out_sp4 = self.conv_out_sp4(feat_res4)
        feat_out_sp8 = self.conv_out_sp8(feat_res8)
        # feat_out_sp16 = self.conv_out_sp16(feat_res16)

        # 语义分支
        feat_cp32 = self.cp(x)

        # 融合结构得到结果
        feat_bga = self.bga(feat_res8, feat_cp32)
        # 分割头处理，恢复原大小，并得到19种分类概率
        logits = self.head(feat_bga)
        # dd torch.Size([4, 19, 512, 1024])
        # print()
        # print("logits.shape")
        # print(logits.shape)

        # 和损失函数对接的内容，包括边缘检测损失和 语义分割损失
        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
            return logits, feat_out_sp2, feat_out_sp4, feat_out_sp8

        if (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
            return logits, feat_out_sp4, feat_out_sp8

        if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
            return logits, feat_out_sp8

        if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
            return logits

    def init_weight(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # self.load_pretrain()

    # def load_pretrain(self):
    #     state = modelzoo.load_url(backbone_url)
    #     for name, child in self.named_children():
    #         if name in state.keys():
    #             child.load_state_dict(state[name], strict=True)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    net = BiSeNet('STDCNet1446', 19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 512, 1024).cuda()
    out = net(in_ten)
    print(out.shape)
    torch.save(net.state_dict(), 'STDCNet1446.pth')


