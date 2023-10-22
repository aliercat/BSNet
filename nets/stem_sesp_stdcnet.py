import torch
import torch.nn as nn
from torch.nn import init
import math


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x



class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


## dd
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

## dd
class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=nn.BatchNorm2d):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)
 
    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))
 
        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))
 
        x = self.relu(x1 + x2)
        x = self.conv3(x).sigmoid()
        # 返回的是注意力程度值，后续还需要逐像素相乘
        return x

## dd
class SESPCatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1, reduction=16):
        super(SESPCatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.se_list = nn.ModuleList()
        self.stride = stride
        self.flag = stride

        #dd
        if stride == 1:
            self.flag = 1
            #se通道应该和convx同步
            for idx in range(block_num):# 变通道
                if idx == 0:
                    self.se_list.append(SELayer(out_planes//2, reduction))
                elif idx == 1 and block_num == 2:
                    self.se_list.append(SELayer(out_planes//2, reduction))
                elif idx == 1 and block_num > 2:
                    self.se_list.append(SELayer(out_planes//4, reduction))
                elif idx < block_num - 1:
                    self.se_list.append(SELayer(out_planes//int(math.pow(2, idx+1)), reduction))
                else:
                    self.se_list.append(SELayer(out_planes//int(math.pow(2, idx)), reduction))
                

        if stride == 2:
            self.flag = 2
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            # 新加SPM
            self.sp = SPBlock(out_planes//2, out_planes//2)

            stride = 1


        for idx in range(block_num):# 只是ConVx变通道
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        if self.flag == 1:# stdc(s=1)
            out=out1
            out_se=self.se_list[0](out)
            out_list.append(out_se)
            # out_list.append(out)

            for idx, conv in enumerate(self.conv_list[1:]):
                out = conv(out)#往后传

                out_se=self.se_list[idx+1](out)#放入合并列表中
                out_list.append(out_se)

        elif self.flag == 2:# stdc(s=2)
            out_sp = self.skip(out1)
            out_sp=out_sp * self.sp(out_sp)
            out_list.append(out_sp)

            for idx, conv in enumerate(self.conv_list[1:]):  
                if idx == 0:#block2
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out)

                out_list.append(out)


        out = torch.cat(out_list, dim=1)
        return out

## dd
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):# 输入通道数，输出通道数，卷积核大小，步长，填充，扩张，这里是same
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
class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 64, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(64, 32, 1, stride=1, padding=0),
            ConvBNReLU(32, 64, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(128, 64, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat

        
#STDC2Net，修改的这个先试试
class STDCNet1446(nn.Module):
    def __init__(self, base=64, layers=[4,5,3], block_num=4, type="sesp", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
        super(STDCNet1446, self).__init__()
        if type == "sesp":
            block = SESPCatBottleneck        
        elif type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck

        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)

        # self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)

        # self.gap = nn.AdaptiveAvgPool2d(1)
        # #FC1
        # self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)

        # self.bn = nn.BatchNorm1d(max(1024, base*16))
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=dropout)
        # #FC2
        # self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

        # self.x2 = nn.Sequential(self.features[:1])
        # self.x4 = nn.Sequential(self.features[1:2])
        # self.x8 = nn.Sequential(self.features[2:6])
        # self.x16 = nn.Sequential(self.features[6:11])
        # self.x32 = nn.Sequential(self.features[11:])

        self.x4 = nn.Sequential(self.features[:1])
        self.x8 = nn.Sequential(self.features[1:5])
        self.x16 = nn.Sequential(self.features[5:10])
        self.x32 = nn.Sequential(self.features[10:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):
        
        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        print("self_state_dict:")
        print(self_state_dict.keys())

        for i in range(2,14):
            prefix="features."+str(i)
            prefix1="features."+str(i-1)
            for k, v in state_dict.items():  
                if k.startswith(prefix):   
                    print(k.replace(prefix, prefix1))
                    self_state_dict.update({k.replace(prefix, prefix1): v})
                    
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        #stage1和2
        #features += [ConvX(3, base//2, 3, 2)]
        #features += [ConvX(base//2, base, 3, 2)]
        features += [StemBlock()]

        for i, layer in enumerate(layers):#[4,5,3].对应stage3-5,且每个stage分别是4，5，3层
            for j in range(layer):
                if i == 0 and j == 0:#stage3的第一层
                    features.append(block(base, base*4, block_num, 2))#base-》base*4，输出通道扩大4倍，变为256，尺寸减半，block=CatBottleneck/AddBottleneck
                elif j == 0:#比如stage4的第一层
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))# base*4-》base*8，通道数翻扩大2倍，尺寸减半
                else:#stage3的第2层
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))#base*4-》base*4，通道数,尺寸不变

        return nn.Sequential(*features)

    # def forward(self, x):# 训练时用，因为还有边缘检测
    #     feat2 = self.x2(x)
    #     feat4 = self.x4(feat2)
    #     feat8 = self.x8(feat4)
    #     feat16 = self.x16(feat8)
    #     feat32 = self.x32(feat16)
    #     if self.use_conv_last:
    #        feat32 = self.conv_last(feat32)

    #     return feat2, feat4, feat8, feat16, feat32

    #def forward_impl(self, x):#推理时用，不用边缘检测
    def forward(self, x):# 边缘检测不在这了，所以不用分 训练和推理了，但是好像不用后面那些分类的地步？
        # out = self.features(x)#就是self._make_layers
        feat4 = self.x4(x)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        #out = self.conv_last(out).pow(2)
        #out = self.gap(out).flatten(1)
        #out = self.fc(out)#也是线性层，FC1024
        # out = self.bn(out)
        #out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        #out = self.dropout(out)
        #out = self.linear(out)#FC1000
        # return out
        return feat4,feat8,feat16,feat32

# STDC1Net，先没改
class STDCNet813(nn.Module):
    def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
        super(STDCNet813, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base*16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):
        
        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
           feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    # model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
    model = STDCNet1446(num_classes=1000, dropout=0.00, pretrain_model='/home/fzl/dfr/my_improved_STDC_copy2/checkpoints/STDCNet1446_76.47.tar',block_num=4)
    
    model.eval()
    #x = torch.randn(1,3,224,224)
    x = torch.randn(1,3,512,1024)
    y = model(x)
    torch.save(model.state_dict(), 'cat.pth')
    print(y.size())
