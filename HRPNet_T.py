import torch
from torch import nn
import torch.nn.functional as F
from DFormer import DFormer_Tiny, DFormer_Base, DFormer_Large
import numpy as np
import cv2
from norm import BasicConv2d, Up, FilterLayer , GloRe_Unit_2D,apply_frequency_filter, DFFM, CoordAtt

class MCCM(nn.Module):
    def __init__(self, inchannel):
        super(MCCM, self).__init__()
        self.c13 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(1, 3), padding=(0, 1))
        self.c31 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(3, 1), padding=(1, 0))
        self.c13_2 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(1, 3), padding=(0, 1))
        self.c31_2 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(3, 1), padding=(1, 0))
        self.aux_conv = FilterLayer(inchannel)

        self.bn1 = nn.BatchNorm2d(inchannel)
        self.sof = nn.Softmax(dim=1)
        self.fuseconv = BasicConv2d(inchannel * 2, inchannel, kernel_size=3, padding=1)
        self.param = torch.nn.Parameter(torch.rand([1, inchannel, 1, 1]), requires_grad=True)

    def forward(self, x, y):
        x1 = self.c13(x)
        x1 = self.c31(x1)
        x2 = self.c31_2(x)
        x2 = self.c13_2(x2)
        f = torch.cat((x1, x2), dim=1)
        f = self.fuseconv(f)

        au = self.aux_conv(y)
        w = au * f
        x1 = w + x1
        x2 = w + x2
        param = torch.sigmoid(self.param)
        f = x1 * param + x2 * (1 - param)
        out = self.sof(self.bn1(f))
        out = out * x
        return out

class fusion(nn.Module):
    def __init__(self, inc):
        super(fusion, self).__init__()
        self.ar = MCCM(inchannel=inc)
        self.sof = nn.Softmax(dim=1)
        self.er = DFFM(inc)
        self.dropout = nn.Dropout(.1)

    def forward(self, r, d):
        br = self.ar(r, d)
        bd = self.ar(d, r)
        br = self.sof(br)
        bd = self.sof(bd)
        br = br * r + r
        bd = bd * d + d
        out = self.er(br, bd)
        out = self.dropout(out)
        return out


class FM(nn.Module):
    def __init__(self, inc):
        super(FM, self).__init__()
        outc = inc * 2
        self.con1 = Up(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.con2 = nn.Conv2d(inc, inc // 2, 1)
        self.fun = fusion(inc // 2)
        self.con3 = BasicConv2d(inc // 2, inc, 1)

    def forward(self, r, d, s):
        r = self.con2(r)  # 128,52
        d = self.con2(d)  # 64,104
        s = self.con1(s)
        r = r + s
        d = d + s
        s = self.fun(r, d)
        s = self.con3(s)
        return s


class FM2(nn.Module):
    def __init__(self, inc=144):
        super(FM2, self).__init__()
        self.con1 = Up(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.con2 = nn.Conv2d(inc, inc // 2, 1)
        self.fun = fusion(inc // 2)
        self.con5 = nn.Conv2d(192, 144, 1)
        self.con3 = BasicConv2d(inc // 2, inc, 1)

    def forward(self, r, d, s):
        r = self.con2(r)
        d = self.con2(d)
        s = self.con1(s)
        s = self.con5(s)
        r = r + s
        d = d + s
        s = self.fun(r, d)
        s = self.con3(s)
        return s


class FM3(nn.Module):
    def __init__(self, inc=144):
        super(FM3, self).__init__()
        self.con1 = Up(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.con2 = nn.Conv2d(inc, inc // 2, 1)
        self.fun = fusion(inc // 2)
        # self.con5 = nn.Conv2d(192, 144, 1)
        self.con3 = BasicConv2d(inc // 2, inc, 1)

    def forward(self, r, d, s):
        r = self.con2(r)
        d = self.con2(d)
        s = self.con1(s)
        r = r + s
        d = d + s
        s = self.fun(r, d)
        s = self.con3(s)
        return s


class GBF(nn.Module):
    def __init__(self, inc):
        super(GBF, self).__init__()

        self.con = BasicConv2d(inc, inc, 3, 1, 1)
        self.con1 = BasicConv2d(inc, inc, 5, 1, 2)
        self.tu = GloRe_Unit_2D(inc, inc, True)  # 图卷积

    def forward(self, r, d):
        r = self.tu(r)
        d = self.tu(d)
        s = r + d
        # 近似拉普拉斯
        s = self.con1(s) - self.con(s)
        d = apply_frequency_filter(d)
        r = s + r
        d = s + d
        return r, d
        
class AA(nn.Module):
    def __init__(self, inc=192):
        super(AA, self).__init__()
        # self.dim = dim
        # self.out_dim = dim
        self.fuse1 = fusion(96)
        self.fuse2 = FM(192)
        self.fuse3 = FM2(288)
        self.fuse4 = FM3(576)
        # self.conv = nn.Conv2d(320, 256, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.Conv43 = nn.Sequential(nn.Conv2d(864, inc, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(True),
                                    nn.Conv2d(inc, inc, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(True))

        self.Conv432 = nn.Sequential(nn.Conv2d(inc * 2, inc, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(inc),
                                     nn.ReLU(True),
                                     nn.Conv2d(inc, inc, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inc),
                                     nn.ReLU(True))

        self.Conv4321 = nn.Sequential(nn.Conv2d(288, inc, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(inc),
                                      nn.ReLU(True),
                                      nn.Conv2d(inc, inc, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(inc),
                                      nn.ReLU(True))

        self.sal_pred = nn.Sequential(nn.Conv2d(inc, 96, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(96),
                                      nn.ReLU(True),
                                      nn.Conv2d(96, 1, 3, 1, 1, bias=False))
        self.linear2 = nn.Conv2d(576, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(192, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(192, 1, kernel_size=3, stride=1, padding=1)

        self.aspp_rgb = CoordAtt(576)
        self.aspp_depth = CoordAtt(576)

        self.tu = GBF(576)
        self.co = nn.Conv2d(576, 192, 1)

    def forward(self, x, feature_list):
        b, c, h, w = feature_list[0].shape
        feature_list0 = torch.cat((feature_list[0], feature_list[0]), dim=1)
        feature_list0 = feature_list0.view(b, 2, c, h, w).permute(1, 0, 2, 3, 4)
        R1, D1 = feature_list0.unbind(dim=0)

        b, c, h, w = feature_list[1].shape
        feature_list1 = torch.cat((feature_list[1], feature_list[1]), dim=1)
        feature_list1 = feature_list1.view(b, 2, c, h, w).permute(1, 0, 2, 3, 4)
        R2, D2 = feature_list1.unbind(dim=0)

        b, c, h, w = feature_list[2].shape
        feature_list2 = torch.cat((feature_list[2], feature_list[2]), dim=1)
        feature_list2 = feature_list2.view(b, 2, c, h, w).permute(1, 0, 2, 3, 4)
        R3, D3 = feature_list2.unbind(dim=0)

        b, c, h, w = feature_list[3].shape
        feature_list3 = torch.cat((feature_list[3], feature_list[3]), dim=1)
        feature_list3 = feature_list3.view(b, 2, c, h, w).permute(1, 0, 2, 3, 4)
        R4, D4 = feature_list3.unbind(dim=0)

        f1 = self.fuse1(R1, D1)
        f2 = self.fuse2(R2, D2, f1)
        f3 = self.fuse3(R3, D3, f2)
        f4 = self.fuse4(R4, D4, f3)

        F3 = self.up2(f4)
        F3 = torch.cat((F3, f3), dim=1)
        F3 = self.Conv43(F3)

        F2 = self.up2(F3)
        F2 = torch.cat((F2, f2), dim=1)
        F2 = self.Conv432(F2)

        F1 = self.up2(F2)
        F1 = torch.cat((F1, f1), dim=1)
        F1 = self.Conv4321(F1)  # [B, 128, 56, 56]
        smap = self.sal_pred(F1)

        T4 = F.interpolate(self.linear2(f4), size=x.size()[2:], mode='bilinear', align_corners=False)
        T3 = F.interpolate(self.linear3(F3), size=x.size()[2:], mode='bilinear', align_corners=False)
        T2 = F.interpolate(self.linear4(F2), size=x.size()[2:], mode='bilinear', align_corners=False)
        T1 = self.up4(smap)
        return T1, T2, T3, T4, f1, f2, f3, f4, F3, F2, F1

class Dfo_T(nn.Module):
    def __init__(self):
        super().__init__()
        self.dformer = DFormer_Large(pretrained=True)
        self.decoder = AA(192)  
        self.conv1_1 = nn.Conv2d(576, 288, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_3 = nn.Conv2d(288, 192, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_5 = nn.Conv2d(192, 96, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_7 = nn.Conv2d(96, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_11 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_1 = nn.Conv2d(576, 288, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_3 = nn.Conv2d(288, 192, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_5 = nn.Conv2d(192, 96, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_7 = nn.Conv2d(96, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_11 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sig = nn.Sigmoid()

    def forward(self, input_rgb, input_depth):
        out = self.dformer(input_rgb, input_depth)

        T1, T2, T3, T4, f1, f2, f3, f4, F3, F2, F1 = self.decoder(input_rgb, out)
        return self.sig(T1),  self.sig(T2),  self.sig(T3),  self.sig(T4)   # f1, f2, f3, f4, F3, F2, F1
