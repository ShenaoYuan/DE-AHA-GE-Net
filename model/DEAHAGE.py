import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model1 import GlobalSelfAttention


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)
    def forward(self, x):
        resout = self.conv1x1(x)
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = resout + out2
        out = self.relu(out3)
        return out

class Decoder(nn.Module):
    def __init__(self, ch_d_in, ch_d_out):
        super(Decoder, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(ch_d_in, ch_d_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_d_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_d_out, ch_d_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_d_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        d_out = self.d_conv(x)
        return d_out

def downsample():
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    return maxpool

class upsampleBlock(nn.Module):
    def __init__(self, up_ch_in, up_ch_out):
        super(upsampleBlock, self).__init__()
        self.upblock = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(up_ch_in, up_ch_out, kernel_size=3, padding=1,bias=True),
            nn.BatchNorm2d(up_ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.upblock(x)
        return x

class AdaptiveChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AdaptiveChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)  # 共享全连接层
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        combined = avg_out + max_out
        combined = self.relu(combined)
        combined = self.fc2(combined)
        scale = self.sigmoid(combined)
        c_atten_out = x * scale + x
        c_out = self.relu(c_atten_out)
        return c_out
class AdaptiveSpatialAttention(nn.Module):
    def __init__(self, kernel_size1=7, kernel_size2=3, dilation=1):
        super(AdaptiveSpatialAttention, self).__init__()
         self.conv1 = nn.Conv2d(2, 1, kernel_size1, padding=(kernel_size1 - 1) // 2, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size2, padding=dilation * ((kernel_size2 - 1) // 2), dilation=dilation,
                               bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat_out = torch.cat([avg_out, max_out], dim=1)
        spatial_scale1 = self.conv1(concat_out)
        spatial_scale1 = self.bn(spatial_scale1)
        spatial_scale1 = F.relu(spatial_scale1)
        spatial_scale2 = self.conv2(spatial_scale1)
        scale = self.sigmoid(spatial_scale2)
        s_atten_out = x * scale
        return s_atten_out

class AdaptiveHybridAttentionMechanism(nn.Module):
    def __init__(self, in_channels=512, reduction=16, kernel_size=7):
        super(AdaptiveHybridAttentionMechanism, self).__init__()
        self.channel_attention = AdaptiveChannelAttention(in_channels, reduction)
        self.spatial_attention = AdaptiveSpatialAttention(kernel_size)
    def forward(self, x):
        x_out_c = self.channel_attention(x)
        x_out_s = self.spatial_attention(x_out_c)
        h_out = x_out_s + x
        return h_out
class GEblock(nn.Module):
    def __init__(self, GE_ch):
        super(GEblock, self).__init__()
        self.conv5x5_x = nn.Conv2d(GE_ch, GE_ch, kernel_size=5, padding=2)
        self.conv7x7_x = nn.Conv2d(GE_ch, GE_ch, kernel_size=7, padding=3)
        self.conv5x5_y = nn.Conv2d(GE_ch, GE_ch, kernel_size=5, padding=2)
        self.conv7x7_y = nn.Conv2d(GE_ch, GE_ch, kernel_size=7, padding=3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_weight_5x5 = nn.Linear(GE_ch, GE_ch)
        self.fc_weight_7x7 = nn.Linear(GE_ch, GE_ch)
        self.bn = nn.BatchNorm2d(GE_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, y):
        b, c, h, w = x.size()
        x1 = self.relu(self.conv5x5_x(x))
        x2 = self.relu(self.conv7x7_x(x))
        y1 = self.conv5x5_y(y)
        y2 = self.conv7x7_y(y)
        m_y1 = self.avg_pool(y1).view(y1.size(0), -1)
        m_y2 = self.avg_pool(y2).view(y2.size(0), -1)
        weight_5x5 = torch.sigmoid(self.fc_weight_5x5(m_y1)).view(-1, y1.size(1), 1, 1)
        weight_7x7 = torch.sigmoid(self.fc_weight_7x7(m_y2)).view(-1, y2.size(1), 1, 1)
        GE = self.relu(self.bn(x1 * weight_7x7 + x2 * weight_5x5 + x))
        return GE

class DEAHAGE(nn.Module):
    def __init__(self, in_ch, n_classes):
        super(DEAHAGE, self).__init__()
        self.n_classes = n_classes
        self.down = downsample()
        self.conv1 = ResBlock(ch_in=in_ch, ch_out=64)
        self.conv2 = ResBlock(ch_in=64, ch_out=128)
        self.conv3 = ResBlock(ch_in=128, ch_out=256)
        self.conv4 = ResBlock(ch_in=256, ch_out=512)
        self.conv5 = ResBlock(ch_in=512, ch_out=1024)
        self.conv6 = ResBlock(ch_in=in_ch, ch_out=64)
        self.conv7 = ResBlock(ch_in=64, ch_out=128)
        self.conv8 = ResBlock(ch_in=128, ch_out=256)
        self.conv9 = ResBlock(ch_in=256, ch_out=512)
        self.conv10 = ResBlock(ch_in=512, ch_out=1024)
        self.AHA = AdaptiveHybridAttentionMechanism(in_channels=1024)
        self.GEblock1 = GEblock(GE_ch=64)
        self.GEblock2 = GEblock(GE_ch=128)
        self.GEblock3 = GEblock(GE_ch=256)
        self.GEblock4 = GEblock(GE_ch=512)
        self.up5 = upsampleBlock(up_ch_in=1024, up_ch_out=512)
        self.up_conv5 = Decoder(ch_d_in=1024, ch_d_out=512)
        self.up4 = upsampleBlock(up_ch_in=512, up_ch_out=256)
        self.up_conv4 = Decoder(ch_d_in=512, ch_d_out=256)
        self.up3 = upsampleBlock(up_ch_in=256, up_ch_out=128)
        self.up_conv3 = Decoder(ch_d_in=256, ch_d_out=128)
        self.up2 = upsampleBlock(up_ch_in=128, up_ch_out=64)
        self.up_conv2 = Decoder(ch_d_in=128, ch_d_out=64)
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.GFE0 = GlobalSelfAttention(in_channels=3,scale=16)
        self.GFE1 = GlobalSelfAttention(in_channels=64,scale=8)
        self.GFE2 = GlobalSelfAttention(in_channels=128,scale=4)
        self.GFE3 = GlobalSelfAttention(in_channels=256,scale=2)
        self.GFE4 = GlobalSelfAttention(in_channels=512,scale=1)
    def forward(self, x):
        x1 = self.conv1(x)
        xd1 = self.down(x1)
        x2 = self.conv2(xd1)
        xd2 = self.down(x2)
        x3 = self.conv3(xd2)
        xd3 = self.down(x3)
        x4 = self.conv4(xd3)
        xd4 = self.down(x4)
        x5 = self.conv5(xd4)
        g = self.GFE0(x)
        g1 = self.GFE1(x1)
        g2 = self.GFE2(x2)
        g3 = self.GFE3(x3)
        g4 = self.GFE4(x4)
        y1 = self.conv6(g) + g1 + x1
        yd1 = self.down(y1)
        y2 = self.conv7(yd1) + g2 + x2
        yd2 = self.down(y2)
        y3 = self.conv8(yd2) + g3 + x3
        yd3 = self.down(y3)
        y4 = self.conv9(yd3) + g4 + x4
        yd4 =self.down(y4)
        y5 = self.conv10(yd4) + x5
        AHA_OUT = self.AHA(y5)
        d5 = self.up5(AHA_OUT)
        ge4 = self.GEblock4(x4, y4)
        c5 = torch.cat(tensors=(d5, ge4), dim=1)
        du5 = self.up_conv5(c5)
        d4 = self.up4(du5)
        ge3 = self.GEblock3(x3, y3)
        c4 = torch.cat(tensors=(d4, ge3), dim=1)
        du4 = self.up_conv4(c4)
        d3 = self.up3(du4)
        ge2 = self.GEblock2(x2, y2)
        c3 = torch.cat(tensors=(d3, ge2), dim=1)
        du3 = self.up_conv3(c3)
        d2 = self.up2(du3)
        ge1 = self.GEblock1(x1, y1)
        c2 = torch.cat(tensors=(d2, ge1), dim=1)
        du2 = self.up_conv2(c2)
        outs = self.conv1x1(du2)
        out = torch.sigmoid(outs)
        return out
