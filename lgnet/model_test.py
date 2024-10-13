import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List
import numpy as np
from mytoken import mytokenize
from mytoken import read_word
import cv2
from model import LGNet
from model import LGNet_L
from model import LGNet_M
from model import LGNet_S
from model import conv_dw
from model import ConvBNReLU
from model import DownConvBNReLU
from model import UpConvBNReLU
from model import RSU



def show_image_shape():
    img = cv2.imread("/home/looksaw/looksawCode/LGNet/lgnet/dataset/LangIR_IRSTD-1k/images/XDU2.png")
    print(img.shape)
    print(img)
    img_mask = cv2.imread("/home/looksaw/looksawCode/LGNet/lgnet/dataset/LangIR_IRSTD-1k/masks/XDU2.png")
    print(img_mask.shape)
    print(img_mask)
def test_LGNet():
    random_imageSize = [2,3,512,512]
    random_imageinput = torch.randn(random_imageSize)
    random_token_size = [3,32]
    random_tokrnInput = torch.randn(random_token_size)
    net = LGNet_S()
    x = net(random_imageinput,random_tokrnInput)
    i = 0
    for xi in x:
        xi = nn.Sigmoid()(xi)
        x_np = xi.detach().numpy()
        max = np.max(x_np)
        print("max {}".format(max))
        print("x{} shape is{}".format(i,xi.shape))
        i += 1
def test_conv_dw():
    random_size = [10,3,32,32]
    random_input = torch.randn(random_size)
    net = conv_dw(inp = 3 ,oup = 12)
    x = net(random_input)
    assert x.shape == torch.Size([10,12,1,1])
    
def test_ConvBNReLU():
    random_size = [10,3,32,32]
    random_input = torch.randn(random_size)
    net = ConvBNReLU(in_ch=  3 ,out_ch = 12)
    x = net(random_input)
    print(x.shape)
    
def test_DownConvBNReLU():
    random_size = [10,3,32,32]
    random_input = torch.randn(random_size)
    net = DownConvBNReLU(in_ch=  3 ,out_ch = 12)
    x = net(random_input)
    print(x.shape)
    
def test_UpConvBNReLU():
    random_size = [10,3,32,32]
    random_input = torch.randn(random_size)
    net = UpConvBNReLU(in_ch=  3 ,out_ch = 12)
    x = net(random_input,random_input)
    print(x.shape)
def test_RSU():
    random_size = [10,3,32,32]
    random_input = torch.randn(random_size)
    net = RSU(height= 6 , in_ch= 3 ,mid_ch = 12,out_ch = 12)
    x = net(random_input,random_input)
    print(x.shape)
    


if __name__ == "__main__":
  test_conv_dw()

     