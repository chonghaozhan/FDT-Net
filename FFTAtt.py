import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Coordinate Attention by the website of CVPR2021|| Coordinate Attention 注意力机制
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
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

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


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class CALayer(nn.Module):  # 通道注意力(original paper)
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fft_mean = img_fft_mean
        mid = max(4, channel//reduction)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, mid, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fft_mean(x)
        # y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


'''
# Network10CA
class CALayer(nn.Module):  # 通道注意力(original paper)
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fft_mean = img_fft_mean
        mid = max(4, channel//reduction)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, mid, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        # y = self.fft_mean(x)
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
'''


def img_fft_mean(img):
    b, c, h, w = img.shape
    crow = h//2
    ccol = h//2
    fft_mean = torch.empty([b, c, 1, 1]).to('cuda')
    for i in range(b):
        for j in range(c):
            img_temp_fft1 = torch.fft.fft2(img[i, j, :, :])
            img_temp_shift = torch.fft.fftshift(img_temp_fft1)
            magnitude = torch.abs(img_temp_shift)
            fft_mean[i, j, 0, 0] = torch.mean(magnitude[crow-50:crow+50, ccol-50:ccol+50])
    return fft_mean


def img_fft_mean1(img):
    b, c, h, w = img.shape
    crow = h//2
    ccol = h//2
    fft_mean = torch.empty([b, c, 1, 1]).to('cuda')
    temp = torch.ones(1).to('cuda')
    temp[0] = math.e
    for i in range(b):
        for j in range(c):
            img_temp_fft1 = torch.fft.fft2(img[i, j, :, :])
            img_temp_shift = torch.fft.fftshift(img_temp_fft1)
            magnitude = torch.abs(img_temp_shift)
            fft_mean[i, j, 0, 0] = temp[0]**(-torch.mean(magnitude[crow-50:crow+50, ccol-50:ccol+50]))
    return fft_mean


class SALayer(nn.Module):#王添加(自注意力:self attention)
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PALayer(nn.Module):  # 像素注意力
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 3, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 3, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y