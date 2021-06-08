# generator

import torch.nn as nn
import torch

def E2L(ic,hc,oc,k,s):
    out = nn.Sequential(nn.Conv2d(in_channels=ic, out_channels=hc, kernel_size=1, bias=False),
                        nn.BatchNorm2d(hc),nn.ReLU(),
                        nn.Conv2d(in_channels=hc, out_channels=oc, kernel_size=1, bias=False),
                        nn.BatchNorm2d(oc),nn.ReLU(),
                        nn.MaxPool2d(kernel_size=k,stride=s))
    return out

def D2L(ic,hc,oc,k,s):
    out = nn.Sequential(nn.ConvTranspose2d(in_channels=ic, out_channels=ic, kernel_size=k, stride=s, bias=False),
                        nn.BatchNorm2d(ic),nn.ReLU(),
                        nn.Conv2d(in_channels=ic, out_channels=hc, kernel_size=1, bias=False),
                        nn.BatchNorm2d(hc),nn.ReLU(),
                        nn.Conv2d(in_channels=hc, out_channels=oc, kernel_size=1, bias=False),
                        nn.BatchNorm2d(oc),nn.ReLU())
    return out

class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.f1 = E2L(3,4,6,3,1)
        self.f2 = E2L(6,18,24,3,1)
        self.f3 = E2L(24,40,48,3,1)
        self.f4 = E2L(48,80,96,3,1)
        self.f5 = E2L(96,160,192,3,1)
        
        self.f6 = D2L(192,160,96,3,1)
        self.f7 = D2L(192,80,48,3,1)
        self.f8 = D2L(96,40,24,3,1)
        self.f9 = D2L(48,18,6,3,1)
        self.f10 = D2L(12,6,3,3,1)
        self.f11 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        
    def forward(self, x):
        # encoder
        y1 = self.f1(x)
        y2 = self.f2(y1)
        y3 = self.f3(y2)
        y4 = self.f4(y3)
        y = self.f5(y4)
        # decoder
        #print(y.shape)
        y = self.f6(y)
        #print(y.shape)
        #print(y4.shape)
        y = self.f7(torch.cat([y,y4], dim=1))
        #print(y.shape)
        #print(y3.shape)
        y = self.f8(torch.cat([y,y3], dim=1))
        #print(y.shape)
        #print(y2.shape)
        y = self.f9(torch.cat([y,y2], dim=1))
        #print(y.shape)
        #print(y1.shape)
        y = self.f10(torch.cat([y,y1], dim=1))
        #print(y.shape)
        y = self.f11(y)
        #print(y.shape)
        return y


class generator_dense(nn.Module):
    def __init__(self):
        super(generator_dense,self).__init__()
        self.pad = nn.ZeroPad2d(1)
        #first layer
        self.f1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5,stride=1)
        #recursive layer
        self.f2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3,stride=1),
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        #A
        self.a1 = nn.Sequential(nn.BatchNorm2d(6),nn.ReLU(),nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3,stride=1))
        self.a2 = nn.Sequential(nn.BatchNorm2d(18),nn.ReLU(),nn.Conv2d(in_channels=18, out_channels=12, kernel_size=3,stride=1))
        self.a3 = nn.Sequential(nn.BatchNorm2d(30),nn.ReLU(),nn.Conv2d(in_channels=30, out_channels=12, kernel_size=3,stride=1))
        self.a4 = nn.Sequential(nn.BatchNorm2d(42),nn.ReLU(),nn.Conv2d(in_channels=42, out_channels=6, kernel_size=1,stride=1),
                                nn.BatchNorm2d(6),nn.ReLU())
        #ls
        self.fls = nn.Sequential(nn.BatchNorm2d(12),nn.ReLU())
        #recursivelayers
        self.f3 = nn.Sequential(nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=63),
                                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1,stride=1))#concat A
        #last layer concat out of 2nd A with f1
        self.f4 = nn.Sequential(nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=3),
                                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1,stride=1, padding=1),
                                nn.BatchNorm2d(3),nn.Sigmoid())
        self.ft = nn.Conv2d(6, 3, 1)
        
    def forward(self, x):
        #f1
        y1 = self.f1(x)
        #A
        y = self.a1(y1)
        c = torch.cat([y1,self.pad(y)],1)
        y = self.a2(c)
        c = torch.cat([c,self.pad(y)],1)
        y = self.a3(c)
        c = torch.cat([c,self.pad(y)],1)
        y2 = self.a4(c)
        #f2
        y = self.f2(y2)
        y = self.fls(y)
        #f3
        y = self.f3(y)
        #A
        y2t = self.ft(y2)
        x = torch.cat([y,y2t],1)
        y = self.a1(x)
        c = torch.cat([x,self.pad(y)],1)
        y = self.a2(c)
        c = torch.cat([c,self.pad(y)],1)
        y = self.a3(c)
        c = torch.cat([c,self.pad(y)],1)
        y = self.a4(c)
        #f4
        y = torch.cat([y,y1],1)
        y = self.f4(y)
        return y
    
    
        
