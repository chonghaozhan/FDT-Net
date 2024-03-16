import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from Inception import inception3
from FFTAtt import CoordAtt, CALayer, SALayer, Get_gradient_nopadding
from pytorch_wavelets import DWTForward,DTCWTForward
from layers import Bottle2neck
import math

'''
This is the model of FDT-Net of GFT stage.
'''

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


def Conv2D(
        in_channels: int, out_channels: int,
        is_seperable: bool = False, has_relu: bool = False,
):
    modules = OrderedDict()
    if is_seperable:
        modules['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    else:
        modules['conv2'] = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=2, padding=2)
    if has_relu:
        modules['relu'] = nn.ReLU()
    return nn.Sequential(modules)


  
class se_block(nn.Module):  
    def __init__(self, input_channel, output_channel,ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(input_channel, input_channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(input_channel // ratio, input_channel, bias=False),
                nn.Sigmoid()
        )
        self.out=nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channel, output_channel, 1, padding=0, bias=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y=x*y
        y=self.out(y)
        return y

class Wave(nn.Module):   # DTWA  
    def __init__(self, channel):
        super(Wave, self).__init__()

        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')

        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):  # 3,320,320
        xl,xH=self.xfm(x)

        y = self.ca(xl)*x
        return y




class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, is_seperable=True, has_relu=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = Conv2D(out_channels, out_channels, is_seperable=True, has_relu=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()
        self.coord = CoordAtt(out_channels, out_channels)

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y+x)))
        y = self.coord(y)
        return y+x

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
            nn.ReLU(inplace=True),
        )
        self.stype = stype
        self.scale = scale
        self.width = width

        self.outlay = nn.Sequential(
            nn.Conv2d(planes * self.expansion, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)


        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        out = self.outlay(out)
        return out





class MidResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1_1 = Conv2D(in_channels, out_channels, is_seperable=True, has_relu=False)
        self.conv1_2 = Conv2D(in_channels, out_channels, is_seperable=False, has_relu=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2_1 = Conv2D(out_channels, out_channels, is_seperable=True, has_relu=False)
        self.conv2_2 = Conv2D(out_channels, out_channels, is_seperable=False, has_relu=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        y1_1 = self.conv1_1(x)
        y1_2 = self.conv1_2(x)
        y1 = torch.add(y1_1, y1_2)
        y1 = self.relu(self.norm1(y1))
        y2_1 = self.conv2_1(y1)
        y2_2 = self.conv2_2(y1)
        y2 = torch.add(y2_1, y2_2)
        y = self.relu(self.norm2(y2))
        return y+x


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv0_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, dilation=1, padding=1)
        self.conv0_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, dilation=2, padding=2)
        self.conv1 = Bottle2neck(out_channels, out_channels)
        self.conv2 = Bottle2neck(out_channels, out_channels)
        self.conv3 = ResBlock(out_channels, out_channels)
        self.relu = nn.ReLU()
        


    def forward(self, x):
        y0_1 = self.conv0_1(x)
        y0_2 = self.conv0_2(x)
        y0 = self.relu(torch.add(y0_1, y0_2))
        y1 = self.conv1(y0)
        y2 = self.conv2(y1+y0)
        y3 = self.conv3(y2+y1+y0)
        return y3


class MidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res1 = MidResBlock(channels, channels)
        self.res2 = MidResBlock(channels, channels)
        self.res3 = MidResBlock(channels, channels)
        self.res4 = MidResBlock(channels, channels)
        self.res5 = MidResBlock(channels, channels)
        self.res6 = MidResBlock(channels, channels)
        self.res7 = Conv2D(channels*3, channels, is_seperable=True, has_relu=True)

    def forward(self, x):
        y1 = self.res1(x)
        y2 = self.res2(y1)
        y3 = self.res3(y2)
        y4 = self.res4(y3+x)
        y5 = self.res5(y4)
        y6 = self.res6(y5)
        y = torch.cat((x, y3, y6), dim=1)
        y = self.res7(y)
        return y


 
class DecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, skip2_channel):
        super().__init__()
        self.conv0 = Conv2D(in_channels, out_channels, is_seperable=True, has_relu=True)
        self.conv1 = Conv2D(out_channels*2+skip2_channel, out_channels, is_seperable=True, has_relu=True)
        self.conv2 = Bottle2neck(out_channels, out_channels)
        self.conv3 = Bottle2neck(out_channels, out_channels)
        self.conv4 = ResBlock(out_channels, out_channels)  #ResBlock
        #self.cbam=cbam_block(out_channels*2+skip2_channel, out_channels) 
        self.wave=Wave(channel=out_channels*2+skip2_channel)
        #self.se=se_block(out_channels*2+skip2_channel,out_channels*2+skip2_channel)

    def forward(self, inputs):
        inp, skip1, skip2 = inputs 
        if skip2.shape[1] != 3:
            skip2 = F.interpolate(skip2, scale_factor=0.5, mode='bilinear')
        y = F.interpolate(inp, scale_factor=2, mode='bilinear')
        y = self.conv0(y)
        y = torch.cat((y, skip1, skip2), dim=1)
        #y=self.se(y)
        y=self.wave(y)
        
        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)
        y4 = self.conv4(y3+y2+y1)
        return y4

    
def wt(vimg): # DTCWT
   xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b') #
   xfm=xfm.cuda()
   Yl,Yh=xfm(vimg)
   Yl1 = torch.nn.functional.interpolate(Yl, scale_factor=0.5) # torch.Size([4, 3, 160, 160])

   Yh1=Yh[0][:,:,0,:,:,0]  # Yh[1]: torch.Size([4, 3, 160, 160])
   Yh2=Yh[0][:,:,1,:,:,0]  # Yh[2]: torch.Size([4, 3, 160, 160])
   Yh3=Yh[0][:,:,2,:,:,0]
   Yh4=Yh[0][:,:,3,:,:,0]
   Yh5=Yh[0][:,:,4,:,:,0]
   Yh6=Yh[0][:,:,5,:,:,0]

   Yh7=Yh[0][:,:,0,:,:,1]
   Yh8=Yh[0][:,:,1,:,:,1]
   Yh9=Yh[0][:,:,2,:,:,1]
   Yh10=Yh[0][:,:,3,:,:,1]
   Yh11=Yh[0][:,:,4,:,:,1]
   Yh12=Yh[0][:,:,5,:,:,1]
   Yh13=torch.cat([Yl1,Yh1,Yh2,Yh3,Yh4,Yh5,Yh6,Yh7,Yh8,Yh9,Yh10,Yh11,Yh12],dim=1) # torch.Size([4, 3, 160, 160])*13  -> torch.Size([4, 3*13, 160, 160])

   return [Yl1,Yh13]  # torch.Size([4, 39, 160, 160])

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = inception3()
        self.input = ConvLayer(in_channels=3, out_channels=16, kernel_size=11, stride=1)
        
        self.calayer = CALayer(channel=16)

        self.enc1 = EncoderStage(in_channels=16, out_channels=32)
        self.enc2 = EncoderStage(in_channels=32+39, out_channels=64)
        self.enc3 = EncoderStage(in_channels=64+39, out_channels=128)
        self.enc4 = EncoderStage(in_channels=128+39, out_channels=256)
        
        
        self.midblock = MidBlock(channels=256)
        
        
        self.dec4 = DecoderStage(in_channels=256, out_channels=128, skip2_channel=64)
        self.dec3 = DecoderStage(in_channels=128, out_channels=64, skip2_channel=32)
        self.dec2 = DecoderStage(in_channels=64, out_channels=32, skip2_channel=16)
        self.dec1 = DecoderStage(in_channels=32, out_channels=16, skip2_channel=3)

        self.output = ConvLayer(in_channels=16, out_channels=3, kernel_size=3, stride=1)
        self.output1 = ConvLayer(in_channels=3, out_channels=3, kernel_size=3, stride=1)

    def forward(self, inp):
        
        inp_2_l, inp_2_H = wt(inp)
        inp_4_l, inp_4_H = wt(inp_2_l)
        inp_6_l, inp_6_H = wt(inp_4_l)
        
        
        
        inc = self.inception(inp)  # 3 320
        input = self.input(inc)  # 16 320
        

        env1 = self.enc1(input)  # 32 160  env1 torch.Size([8, 32, 160, 160])
        env2 = self.enc2(torch.cat((env1, inp_2_H), dim=1))  # 64 80
        env3 = self.enc3(torch.cat((env2, inp_4_H), dim=1))  # 128 40
        env4 = self.enc4(torch.cat((env3, inp_6_H), dim=1))  # 256 20
        
        mid = self.midblock(env4)  # 256 20
        dev4 = self.dec4((mid, env3, env2))  # 256->128 40   mid:torch.Size([8, 256, 20, 20]) env3:torch.Size([8, 128, 40, 40]) env2: torch.Size([8, 64, 80, 80])
        
        dev3 = self.dec3((dev4, env2, env1))  # 128->64 80
        
        dec2 = self.dec2((dev3, env1, input))  # 64->32 160
        dec1 = self.dec1((dec2, input, inc))  # 32->16 320

        output = self.output(dec1)  # 16->3 320
        pred = self.output1(output)
        return pred+inp

if __name__ == '__main__':
    x=torch.rand([8,3,320,320]).to('cuda')
    net=Network().to('cuda')
    y=net(x)
    print(y.shape)
