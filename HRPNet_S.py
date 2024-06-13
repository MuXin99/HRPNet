import torch
from torch import nn
import torch.nn.functional as F
from Mirror_ss.backbone.DFormer.DFormer import DFormer_Tiny
from Mirror_ss.Work1.norm import DFFM, BasicConv2d, Up

class FM(nn.Module):
    def __init__(self, inc):
        super(FM, self).__init__()
        self.con1 = Up(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.con2 = nn.Conv2d(inc, inc//2, 1)
        self.fun = DFFM(inc//2)
        self.con3 = BasicConv2d(inc//2, inc, 1)
    def forward(self, r, d, s):
        r = self.con2(r) # 128,52
        d = self.con2(d) # 64,104
        s = self.con1(s)
        r = r + s
        d = d + s
        s = self.fun(r, d)
        s = self.con3(s)
        return s
####################################################################################
class AA(nn.Module):
    def __init__(self, inc=64):
        super(AA, self).__init__()
        self.fuse1 = DFFM(32)
        self.fuse2 = FM(64)
        self.fuse3 = FM(128)
        self.fuse4 = FM(256)
        #self.conv = nn.Conv2d(160, 128, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.Conv43 = nn.Sequential(nn.Conv2d(inc*6, inc, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(True),
                                    nn.Conv2d(inc, inc, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(True))

        self.Conv432 = nn.Sequential(nn.Conv2d(2 * inc, inc, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(True),
                                    nn.Conv2d(inc, inc, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(True))

        self.Conv4321 = nn.Sequential(nn.Conv2d(96, inc, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(True),
                                    nn.Conv2d(inc, inc, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(True))

        self.sal_pred = nn.Sequential(nn.Conv2d(inc, 32, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(True),
                                      nn.Conv2d(32, 1, 3, 1, 1, bias=False))
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x,  feature_list):
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

#################################################################################################################
class Dfo_S(nn.Module):
    def __init__(self):
        super().__init__()
        self.dformer = DFormer_Tiny(pretrained=True)
        self.decoder = AA(64)
        self.fuse_canny_edge = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        # ------------------------  rgb prediction module  ---------------------------- #
        self.conv1_1 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_3 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_5 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_7 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_9 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_11 = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(1, 1))

        # ------------------------  t prediction module  ---------------------------- #
        self.conv2_1 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_3 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_5 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_7 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_9 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_11 = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(1, 1))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sig = nn.Sigmoid()

    def forward(self, input_rgb,input_depth):
        out = self.dformer(input_rgb, input_depth)
        T1, T2, T3, T4, f1, f2, f3, f4, F3, F2, F1 = self.decoder(input_rgb, out)
        return self.sig(T1),  self.sig(T2),  self.sig(T3),  self.sig(T4) #,f1, f2, f3, f4, F3, F2, F1


if __name__ == '__main__':
    img = torch.randn(1, 3, 416, 416)
    depth = torch.randn(1, 3, 416, 416)

    model = Dfo_S()
# 10.67G   13.64M
    out = model(img,depth)
    for i in range(len(out)):
        print(out[i].shape)

    # # from toolbox import compute_speed
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     net = Dfo_S()
    #     flops, params = get_model_complexity_info(net, (3, 416, 416), as_strings=True, print_per_layer_stat=False)
    #     print('Flops: ' + flops)
    #     print('Params: ' + params)

    # compute_speed(net, input_size=(1, 3, 416, 416), iteration=500)