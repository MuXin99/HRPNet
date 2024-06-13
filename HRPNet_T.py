import torch
from torch import nn
import torch.nn.functional as F
from Mirror_ss.backbone.DFormer.DFormer import DFormer_Tiny, DFormer_Base, DFormer_Large
import numpy as np
import cv2
from Mirror_ss.Work1.norm import ChannelAttention, SpatialAttention, BasicConv2d, Up, FilterLayer, DualGCN, \
    GloRe_Unit_2D,apply_frequency_filter


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=96):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class AR(nn.Module):#AngleRefinementModule
    def __init__(self, inchannel):
        super(AR, self).__init__()
        self.c13 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(1, 3), padding=(0, 1))
        self.c31 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(3, 1), padding=(1, 0))
        self.c13_2 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(1, 3), padding=(0, 1))
        self.c31_2 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(3, 1), padding=(1, 0))
        self.aux_conv = FilterLayer(inchannel)

        self.bn1 = nn.BatchNorm2d(inchannel)
        self.sof = nn.Softmax(dim=1)
        self.fuseconv = BasicConv2d(inchannel * 2, inchannel, kernel_size=3, padding=1)

        # self.conv_end = BasicConv2d(inchannel, inchannel, kernel_size=3, padding=1)

        self.param = torch.nn.Parameter(torch.rand([1, inchannel, 1, 1]), requires_grad=True)

    def forward(self, x, y):
        x1 = self.c13(x)
        x1 = self.c31(x1)
        x2 = self.c31_2(x)
        x2 = self.c13_2(x2)
        fuse_max = torch.cat((x1, x2), dim=1)
        fuse_max = self.fuseconv(fuse_max)

        aux_w = self.aux_conv(y)
        weight = aux_w * fuse_max
        x1 = weight + x1
        x2 = weight + x2
        param = torch.sigmoid(self.param)
        f = x1 * param + x2 * (1 - param)
        ar_out = self.sof(self.bn1(f))
        ar_out = ar_out * x
        return ar_out


class Attention(nn.Module):
    def __init__(self, in_x, reduction=16):
        super(Attention, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_x, in_x, 3, padding=1),
            nn.BatchNorm2d(in_x),
            nn.LeakyReLU())
        in_x = in_x * 2
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_x)

        # self.w1 = torch.nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, r, d):
        r = self.down_conv(r)
        d = self.down_conv(d)
        mul_rd = r * d
        sa = self.sa(mul_rd)
        r_f = r * sa
        r_f = r + r_f
        r_ca = self.ca(r_f)
        r_out = r * r_ca
        return r_out


class ER(nn.Module):#DualFeatureFusionModule
    def __init__(self, in_x):
        super(ER, self).__init__()
        down_x = in_x
        self.A = Attention(in_x)
        in_x = in_x * 2
        self.c = BasicConv2d(in_x, in_x // 2, 1)

        self.conv_c = nn.Conv2d(in_x // 2, 1, 1)
        self.conv_c_d = nn.Conv2d(in_x // 2, 1, 1)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_x, in_x, 3, 1, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_x), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_x, in_x, 3, 1, 4, 4, bias=False),
                                     nn.BatchNorm2d(in_x), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_x, in_x, 3, 1, 8, 8, bias=False),
                                     nn.BatchNorm2d(in_x), nn.LeakyReLU(0.1, inplace=True))

        self.b_1 = BasicConv2d(in_x * 3, in_x, kernel_size=3, padding=1)  # CBR33
        self.conv_res = BasicConv2d(in_x, in_x, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def aspp(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))  # fu
        out = self.relu(buffer_1 + self.conv_res(x))  # CAi
        return out

    def forward(self, r, d):
        b, c, h, w = r.shape
        alpha = torch.sum(torch.abs(torch.full([b, c, h, w], 0.5).cuda() - self.sig(self.conv_c(r))))
        beta = torch.sum(torch.abs(torch.full([b, c, h, w], 0.5).cuda()- self.sig(self.conv_c_d(d))))
        alpha = alpha / (alpha + beta)
        beta = 1.0 - alpha
        r_out = self.A(r, d)  # 64,96
        d_out = self.A(d, r)

        # mul_fea = r_out * d_out  # 64,96
        # add_fea = r_out + d_out
        RD = torch.cat([alpha* r_out,  beta*d_out], dim=1)  # 512,13

        out1 = self.aspp(RD)  # 32 48
        out1 = self.c(out1)
        return out1


class fusion(nn.Module):
    def __init__(self, inc):
        super(fusion, self).__init__()
        self.ar = AR(inchannel=inc)
        self.sof = nn.Softmax(dim=1)
        self.er = ER(inc)
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


class BM(nn.Module):
    def __init__(self, inc):
        super(BM, self).__init__()
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


class BM2(nn.Module):
    def __init__(self, inc=144):
        super(BM2, self).__init__()
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


class BM3(nn.Module):
    def __init__(self, inc=144):
        super(BM3, self).__init__()
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


class tu(nn.Module):
    def __init__(self, inc):
        super(tu, self).__init__()

        self.con = BasicConv2d(inc, inc, 3, 1, 1)
        self.con1 = BasicConv2d(inc, inc, 5, 1, 2)
        self.tu = GloRe_Unit_2D(inc, inc, True)  # 图卷积

    def forward(self, r, d):
        r = self.tu(r)
        d = self.tu(d)
        s = r + d
        # 计算拉普拉斯
        s = self.con1(s) - self.con(s)
        d = apply_frequency_filter(d)
        r = s + r
        d = s + d
        return r, d


####################################################################################
class Decoder(nn.Module):
    def __init__(self, inc=192):
        super(Decoder, self).__init__()
        # self.dim = dim
        # self.out_dim = dim
        self.fuse1 = fusion(96)
        self.fuse2 = BM(192)
        self.fuse3 = BM2(288)
        self.fuse4 = BM3(576)
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
        # self.CAtt = CAtt(96)
        self.tu = tu(576)
        self.co = nn.Conv2d(576, 192, 1)

    def forward(self, x, feature_list, feature_list_depth):
        R1, R2, R3, R4 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]
        D1, D2, D3, D4 = feature_list_depth[0], feature_list_depth[1], feature_list_depth[2], feature_list_depth[3]
        #
        R4 = self.aspp_rgb(R4)
        D4 = self.aspp_depth(D4)
        R4, D4 = self.tu(R4, D4)

        RD1 = self.fuse1(R1, D1)
        RD2 = self.fuse2(R2, D2, RD1)  # 1,192,52,52
        RD3 = self.fuse3(R3, D3, RD2)
        RD4 = self.fuse4(R4, D4, RD3)

        egd = self.co(R4 + D4)

        RD43 = self.up2(RD4)
        RD43 = torch.cat((RD43, RD3), dim=1)
        RD43 = self.Conv43(RD43)

        De43 = self.up2(egd) * RD43 + RD43

        RD432 = self.up2(RD43)
        RD432 = torch.cat((RD432, RD2), dim=1)
        RD432 = self.Conv432(RD432)

        De432 = self.up2(De43) * RD432 + RD432

        RD4321 = self.up2(RD432)
        RD4321 = torch.cat((RD4321, RD1), dim=1)
        RD4321 = self.Conv4321(RD4321)  # [B, 128, 56, 56]

        De4321 = self.up2(De432) * RD4321 + RD4321
        De4321 = self.up4(self.sal_pred(De4321))
        sal_map = self.sal_pred(RD4321)
        sal_out = self.up4(sal_map)

        mask4 = F.interpolate(self.linear2(RD4), size=x.size()[2:], mode='bilinear', align_corners=False)
        mask3 = F.interpolate(self.linear3(RD43), size=x.size()[2:], mode='bilinear', align_corners=False)
        mask2 = F.interpolate(self.linear4(RD432), size=x.size()[2:], mode='bilinear', align_corners=False)
        return sal_out, mask4, mask3, mask2, De4321, R1, R2, R3, R4, D1, D2, D3, D4, RD1, RD2, RD3, RD4, RD43, RD432, RD4321


#################################################################################################################
class Dfo_T(nn.Module):
    def __init__(self):
        super().__init__()
        self.dformer = DFormer_Large(pretrained=True)
        self.decoder = Decoder(192)  #
        # ------------------------  rgb prediction module  ---------------------------- #
        self.conv1_1 = nn.Conv2d(576, 288, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_3 = nn.Conv2d(288, 192, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_5 = nn.Conv2d(192, 96, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_7 = nn.Conv2d(96, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_11 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

        # ------------------------  t prediction module  ---------------------------- #
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

        sal1 = self.conv1_11(self.upsample(self.conv1_9(self.upsample(self.conv1_7(
            self.upsample(self.conv1_5(
                self.upsample(self.conv1_3(
                    self.upsample(self.conv1_1(out[3])))))))))))
        sal2 = self.conv2_11(self.upsample(self.conv2_9(self.upsample(self.conv2_7(
            self.upsample(self.conv2_5(
                self.upsample(self.conv2_3(
                    self.upsample(self.conv2_1(out[3])))))))))))
        result_final, mask4, mask3, mask2, De4321, R1, R2, R3, R4, D1, D2, D3, D4, RD1, RD2, RD3, RD4, RD43, RD432, RD4321 = self.decoder(
            input_rgb, out, out)
        return self.sig(result_final), self.sig(mask4), self.sig(mask3), self.sig(mask2), self.sig(De4321), self.sig(sal1), self.sig(sal2), \
            self.sig(R1), self.sig(R2), self.sig(R3), self.sig(R4), self.sig(D1), self.sig(D2), self.sig(D3), self.sig(D4), \
            self.sig(RD1), self.sig(RD2), self.sig(RD3), self.sig(RD4), self.sig(RD43), self.sig(RD432), self.sig(RD4321)
        # 0-6  7-14 15-18 19-21


if __name__ == '__main__':
    img = torch.randn(1, 3, 416, 416)
    depth = torch.randn(1, 3, 416, 416)

    model = Dfo_T()
    # for i in range(len(ouout = model(img,depth)t)):
    #     print(out[i].shape)

    out = model(img, depth)

    # # from toolbox import compute_speed
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     net = Dfo_T()
    #     flops, params = get_model_complexity_info(net, (3, 416, 416), as_strings=True, print_per_layer_stat=False)
    #     print('Flops: ' + flops)
    #     print('Params: ' + params)

    # compute_speed(net, input_size=(1, 3, 416, 416), iteration=500)