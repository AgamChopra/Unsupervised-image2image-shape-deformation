# Discriminator

import torch.nn as nn
import torch
from torchvision import models

#print(dir(models))

class discriminator_resnet(nn.Module):
    def __init__(self):
        super(discriminator_resnet, self).__init__()
        #self.f1 = models.wide_resnet50_2()
        self.f1 = models.resnext101_32x8d(pretrained=True)
        self.f2 = nn.Sequential(nn.Linear(1000,1),nn.Sigmoid())
        
    def forward(self, x):
        y = self.f1(x)
        y = self.f2(y)
        return y

def conv1_layer(in_c):
    conv = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=1, kernel_size = 1, stride = 1),nn.Sigmoid())
    return conv

def conv2_layer(in_c, out_c):
    conv = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size = 2, stride = 2),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2, stride=1))
    return conv

def conv3_layer(in_c, out_c):
    conv = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size = 3, stride = 1),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=3, stride=3))
    return conv

def conv5_layer(in_c, out_c):
    conv = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size = 5, stride = 3),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=3, stride=3))
    return conv

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.f1 = conv5_layer(3, 6)
        self.f2 = conv3_layer(6, 12)
        self.f3 = conv2_layer(12, 24)
        self.f4 = conv1_layer(24)
        
    def forward(self, x):
        y = self.f1(x)
        y = self.f2(y)
        y = self.f3(y)
        y = self.f4(y)
        return y

class discriminator2(nn.Module):
    def __init__(self):
        super(discriminator2, self).__init__()
        self.f1 = conv5_layer(3, 12)
        self.f2 = conv3_layer(12, 48)
        self.f3 = conv2_layer(48, 192)
        self.f4 = conv1_layer(192)
        
    def forward(self, x):
        y = self.f1(x)
        y = self.f2(y)
        y = self.f3(y)
        y = self.f4(y)
        return y

class discriminator3(nn.Module):
    def __init__(self):
        super(discriminator3, self).__init__()
        self.f1 = conv5_layer(3, 128)
        self.f2 = conv3_layer(128, 256)
        self.f3 = conv2_layer(256, 512)
        self.f4 = conv1_layer(512)
        
    def forward(self, x):
        y = self.f1(x)
        y = self.f2(y)
        y = self.f3(y)
        y = self.f4(y)
        return y

def conv_block1(in_c,out_c):
    out = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size = 3, stride = 1),nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size = 2, stride = 2),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=3, stride=1))
    return out

def conv_block2(in_c,out_c):
    out = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size = 3, stride = 1),nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size = 3, stride = 1),nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size = 3, stride = 1),nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size = 2, stride = 1),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=3, stride=1))
    return out

def conv_block3(in_c):
    out = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=1, kernel_size = 1, stride = 1),nn.Sigmoid())
    return out

class discriminator4(nn.Module):
    def __init__(self):
        super(discriminator4, self).__init__()
        self.f1 = conv_block1(3, 6)
        self.f2 = conv_block1(6, 12)
        self.f3 = conv_block1(12, 24)
        self.f4 = conv_block2(24, 48)
        self.f5 = conv_block3(48)
        
    def forward(self, x):
        #print(x.shape)
        y = self.f1(x)
        #print('f1',y.shape)
        y = self.f2(y)
        #print('f2',y.shape)
        y = self.f3(y)
        #print('f3',y.shape)
        y = self.f4(y)
        #print('f4',y.shape)
        y = self.f5(y)
        #print('f5',y.shape)
        return y
    
class discriminator5(nn.Module):
    def __init__(self):
        super(discriminator5, self).__init__()
        self.f1 = conv_block1(3, 12)
        self.f2 = conv_block1(12, 24)
        self.f3 = conv_block1(24, 48)
        self.f4 = conv_block2(48, 96)
        self.f5 = conv_block3(96)
        
    def forward(self, x):
        #print(x.shape)
        y = self.f1(x)
        #print('f1',y.shape)
        y = self.f2(y)
        #print('f2',y.shape)
        y = self.f3(y)
        #print('f3',y.shape)
        y = self.f4(y)
        #print('f4',y.shape)
        y = self.f5(y)
        #print('f5',y.shape)
        return y
    
class discriminator_dense(nn.Module):
    def __init__(self):
        super(discriminator_dense,self).__init__()
        self.f1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=8,stride=2),
                                nn.Conv2d(in_channels=6, out_channels=12, kernel_size=2,stride=1))
        #A1
        self.f2 = nn.Sequential(nn.BatchNorm2d(12),nn.ReLU(),nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3,stride=1))
        self.f3 = nn.Sequential(nn.BatchNorm2d(36),nn.ReLU(),nn.Conv2d(in_channels=36, out_channels=72, kernel_size=3,stride=1))
        self.f4 = nn.Sequential(nn.BatchNorm2d(108),nn.ReLU(),nn.Conv2d(in_channels=108, out_channels=12, kernel_size=1,stride=1),
                                nn.BatchNorm2d(12),nn.ReLU())
        
        self.f5 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5,stride=3,padding=1),
                                nn.MaxPool2d(kernel_size=3, stride=1))
        #A2
        self.f6 = nn.Sequential(nn.BatchNorm2d(24),nn.ReLU(),nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3,stride=1))
        self.f7 = nn.Sequential(nn.BatchNorm2d(72),nn.ReLU(),nn.Conv2d(in_channels=72, out_channels=144, kernel_size=3,stride=1))
        self.f8 = nn.Sequential(nn.BatchNorm2d(216),nn.ReLU(),nn.Conv2d(in_channels=216, out_channels=12, kernel_size=1,stride=1),
                                nn.BatchNorm2d(12),nn.ReLU())
        
        self.f9 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5,stride=3,padding=1),
                                nn.MaxPool2d(kernel_size=3, stride=1),nn.BatchNorm2d(24),nn.ReLU(),
                                nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3,stride=1),
                                nn.MaxPool2d(kernel_size=2, stride=1),nn.BatchNorm2d(48),nn.ReLU(),
                                nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1,stride=1),nn.Sigmoid())
    def forward(self, x):
        x = self.f1(x)
        y = self.f2(x)
        pad = nn.ZeroPad2d(1)
        y = pad(y)
        cat = torch.cat([x,y],1)
        y = self.f3(cat)
        y = pad(y)
        cat = torch.cat([cat,y],1)
        y = self.f4(cat)
        x = self.f5(y)
        y = self.f6(x)
        y = pad(y)
        cat = torch.cat([x,y],1)
        y = self.f7(cat)
        y = pad(y)
        cat = torch.cat([cat,y],1)
        y = self.f8(cat)
        y = self.f9(y)
        return y
    
 #!!!! 
def E2L(ic,hc,oc,k,s):
    out = nn.Sequential(nn.Conv2d(in_channels=ic, out_channels=hc, kernel_size=1),
                        nn.BatchNorm2d(hc),nn.ReLU(),
                        nn.Conv2d(in_channels=hc, out_channels=oc, kernel_size=1),
                        nn.BatchNorm2d(oc),nn.ReLU(),
                        nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=1),
                        nn.BatchNorm2d(oc),nn.ReLU(),
                        nn.MaxPool2d(kernel_size=k,stride=s))
    return out
      
class discriminator_encoder(nn.Module):
    def __init__(self):
        super(discriminator_encoder,self).__init__()
        self.f1 = E2L(3,4,6,5,3)
        self.f2 = E2L(6,18,24,4,2)
        self.f3 = E2L(24,40,48,4,2)
        self.f4 = E2L(48,80,96,3,2)
        self.f5 = E2L(96,160,192,3,1)
        self.f6 = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=384, kernel_size=2),
                                nn.BatchNorm2d(384),nn.ReLU(),
                                nn.Conv2d(in_channels=384, out_channels=1, kernel_size=1),
                                nn.Sigmoid())
    def forward(self, x):
        y = self.f1(x)
        y = self.f2(y)
        y = self.f3(y)
        y = self.f4(y)
        y = self.f5(y)
        y = self.f6(y)
        return y