from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, ceil, floor
import numpy as np


class conv_dw_b(nn.Module):
    def __init__(self, inp, oup): 
        super(conv_dw_b, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64,128, bias=False),
            nn.ReLU(inplace=True),
        )
        self.dwc = nn.AdaptiveAvgPool2d(1)
        self.dwc1 = nn.Conv2d(2*inp, oup, 1, 1, 0, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)         
        self.sig2 = nn.Sigmoid()             
    def forward(self, x, emb): 
        x = self.dwc(x)
        y = self.fc(emb)
        tensor1 = x 
        tensor2 = y
        stacked_tensors = torch.stack((tensor1.view(-1), tensor2.view(-1)), dim=0)
        alternating_tensor = stacked_tensors.transpose(1, 0).reshape(-1).view(x.shape[0],2*x.shape[1],x.shape[3],x.shape[3])
        x = self.bn1(self.dwc1(alternating_tensor))        
        x = self.sig2(x)
        return x





class conv_dw(nn.Module):
    def __init__(self, inp, oup,):
        super(conv_dw, self).__init__()
        self.dwc = nn.AdaptiveAvgPool2d(1) 
        self.dwc1 = nn.Conv2d(inp, oup, 1, 1, 0, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)         
        self.sig2 = nn.Sigmoid()             
    def forward(self, x): 
        # print(self.dwc(x))
        x = self.bn1(self.dwc1(self.dwc(x)))        
        x = self.sig2(x)
        return x
    

class ConvBNReLU(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    """ Residual U-block """
    '''
    height 是模型的高度，in_ch 是输入通道数，mid_ch 是中间通道数，out_ch 是输出通道数
    '''
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)  # stem
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))

        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class fusion(nn.Module):

    def __init__(self, in_channel, kernel_size=1, stride=1):
        super(fusion, self).__init__()

        self.in_channel = in_channel
        self.out_channel = in_channel * 2

        self.out = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 3, 1, 1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(True)
        )

        self.calib1 = conv_dw(in_channel, in_channel) # CCM for conv1
        self.calib1_b = conv_dw_b(in_channel, in_channel)

    def channel(self, E, D, ind, emb):

        if ind ==1:
            y = self.calib1_b(D,emb)        
            out = D*y 
            return out
        else:
            y = self.calib1(D)     
            out = D*y 
            return out

    def spatial(self, E, D, ind, emb):
        if ind ==1:
            y = self.calib1_b(E,emb)        
            out = E*y 
            return out
        else:
            y = self.calib1(E)        
            out = E*y 
            return out

    def forward(self, E, D, emb):
        if E.shape[1] ==128 and E.shape[2]== 32:
            ind = 1
        else:
            ind = 0


        channel = self.channel(E, D, ind, emb)  # [N, C, H, W]
        spatial = self.spatial(E, D, ind, emb)  # [N, C, H, W]
        out = self.out(spatial + channel)  # [N, C, H, W]
        return out


class CBR_new(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_layer = torch.nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=3, stride=1, padding=1, bias=False,dtype=torch.float, device='cuda')
            self.bn = torch.nn.BatchNorm2d(out_channels,dtype=torch.float, device='cuda')

            self.relu = torch.nn.ReLU(inplace=True)
        
        
        def forward(self, x):
            x = self.conv_layer(x)
            x = self.bn(x)          
            x = self.relu(x)
            return x



class LGNet(nn.Module):

    def __init__(self, cfg: dict, out_ch: int = 1, ob=True):
        super().__init__()
        assert "encode" and "decode" in cfg
        self.encode_num = len(cfg["encode"])
        self.ob = ob
        encode_list = []
        loss_list = []
        side_list = []
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) >= 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                if self.ob:
                    loss_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        decode_list = []
        fusion_list = []
        for i, c in enumerate(cfg["decode"]):
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) >= 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            fusion_list.append(fusion(int(c[1] / 2)))

            if c[5] is True:
                if self.ob:
                    loss_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
                    side_list.append(nn.Conv2d(c[3], 2 ** i, kernel_size=3, padding=1))
                    self.side543 = nn.Conv2d(c[3], 1, kernel_size=3, padding=1)
                    self.side543_cat = nn.Conv2d(3, 1, kernel_size=1)
                else:
                    side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))

        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        if self.ob:
            self.loss_modules = nn.ModuleList(loss_list)
            self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)
        else:
            self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

        self.fusion_modules = nn.ModuleList(fusion_list)
        self.bn = nn.BatchNorm2d(out_ch)
        self.linear_layer_emb = nn.Linear(32, 32)
        # Example parameters, adjust as needed
    

    def forward(self, x: torch.Tensor, des: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape


        encode_outputs = []
        filter_list = [64,64,64,128,128,128]
        
        total_null_values = 0

        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        x = encode_outputs.pop()

        des2 = self.linear_layer_emb(des)
        decode_outputs = [x]
        for m in zip(self.decode_modules, self.fusion_modules):
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m[1](x, x2, des2)  
            x = m[0](x)
            decode_outputs.insert(0, x)

        side_outputs = []
        loss_outputs = []

        # ob
        if self.ob:
            y3 = F.interpolate(self.side543(decode_outputs[2]), size=[512, 512], mode='bilinear', align_corners=False)
            y4 = F.interpolate(self.side543(decode_outputs[1]), size=[512, 512], mode='bilinear', align_corners=False)
            y5 = self.side543(decode_outputs[0])
            y = self.side543_cat(torch.concat([y3, y4, y5], dim=1))
            y = torch.sigmoid(y)
            for i, m in enumerate(zip(self.side_modules, self.loss_modules)):
                x = decode_outputs.pop()
                z = x.clone()
                z = F.interpolate(m[1](z), size=[h, w], mode='bilinear', align_corners=False)
                if i <= 2:
                    x = F.interpolate(m[0](x), size=[h, w], mode='bilinear', align_corners=False)
                    x = x * y
                else:
                    x = F.interpolate(m[0](x), size=[h, w], mode='bilinear', align_corners=False)
                side_outputs.insert(0, x)
                loss_outputs.insert(0, z)
        else:
            for m in self.side_modules:
                x = decode_outputs.pop()
                x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
                side_outputs.insert(0, x)

        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            if self.ob:
                return [x] + loss_outputs
            else:
                return [x] + side_outputs
        else:
            return self.bn(x)


def LGNet_L(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 32, 64, False, False],  # En3
                   [4, 64, 32, 128, False, False],  # En4
                   [4, 128, 32, 128, True, False],  # En5
                   [4, 128, 64, 128, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 256, 64, 128, True, True],  # De5
                   [4, 256, 32, 64, False, True],  # De4
                   [5, 128, 32, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return LGNet(cfg, out_ch, ob=False)


def LGNet_M(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return LGNet(cfg, out_ch)


def LGNet_S(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 4, 8, False, False],  # En1
                   [6, 8, 4, 8, False, False],  # En2
                   [5, 8, 4, 8, False, False],  # En3
                   [4, 8, 4, 8, False, False],  # En4
                   [4, 8, 4, 8, True, False],  # En5
                   [4, 8, 4, 8, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 16, 4, 8, True, True],  # De5
                   [4, 16, 4, 8, False, True],  # De4
                   [5, 16, 4, 8, False, True],  # De3
                   [6, 16, 4, 8, False, True],  # De2
                   [7, 16, 4, 8, False, True]]  # De1
    }

    return LGNet(cfg, out_ch)
