import torch
import torch.nn as nn
import torch.nn.functional as F

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(XceptionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        residual = x
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        shortcut = self.shortcut(residual)
        x = x + shortcut
        return x

class EntryFlow(nn.Module):
    def __init__(self, in_channels):
        super(EntryFlow, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.block1 = XceptionBlock(64, 128)
        self.block2 = XceptionBlock(128, 256)
        self.block3 = XceptionBlock(256, 728)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class MiddleFlow(nn.Module):
    def __init__(self):
        super(MiddleFlow, self).__init__()
        self.blocks = nn.Sequential(
            *[XceptionBlock(728, 728) for _ in range(8)]
        )
        
    def forward(self, x):
        return self.blocks(x)

class ExitFlow(nn.Module):
    def __init__(self):
        super(ExitFlow, self).__init__()
        self.block = XceptionBlock(728, 1024)
        self.conv1 = nn.Conv2d(1024, 1536, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1536, 2048, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        x = self.block(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class PixelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttentionModule, self).__init__()
        self.conv_q = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        q = self.conv_q(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        k = self.conv_k(x).view(batch_size, -1, H*W)
        v = self.conv_v(x).view(batch_size, -1, H*W)
        
        attention = torch.bmm(q, k)
        attention = self.softmax(attention)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C//8, H, W)
        out = out + x
        return out

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.conv_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        q = self.conv_q(x).view(batch_size, C, -1)
        k = self.conv_k(x).view(batch_size, C, -1).permute(0, 2, 1)
        v = self.conv_v(x).view(batch_size, C, -1)
        
        attention = torch.bmm(q, k)
        attention = self.softmax(attention)
        out = torch.bmm(attention, v)
        out = out.view(batch_size, C, H, W)
        out = out + x
        return out

class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionModule, self).__init__()
        self.pam = PixelAttentionModule(in_channels)
        self.cam = ChannelAttentionModule(in_channels)
        
    def forward(self, x):
        out_pam = self.pam(x)
        out_cam = self.cam(x)
        return out_pam + out_cam

class DeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.deconv1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv2(x)
        return x

class AXUNet(nn.Module):
    def __init__(self):
        super(AXUNet, self).__init__()
        self.entry_flow = EntryFlow(3)
        self.middle_flow = MiddleFlow()
        self.exit_flow = ExitFlow()
        self.attention1 = SelfAttentionModule(728)
        self.attention2 = SelfAttentionModule(728)
        self.attention3 = SelfAttentionModule(728)
        self.attention4 = SelfAttentionModule(728)
        self.deblock1 = DeBlock(728, 256)
        self.deblock2 = DeBlock(256, 128)
        self.deblock3 = DeBlock(128, 64)
        self.deblock4 = DeBlock(64, 32)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        
    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.attention1(x)
        x = self.deblock1(x)
        x = self.attention2(x)
        x = self.deblock2(x)
        x = self.attention3(x)
        x = self.deblock3(x)
        x = self.attention4(x)
        x = self.deblock4(x)
        x = self.final_conv(x)
        return x


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target):
        smooth = 1.0
        pred = torch.sigmoid(pred)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return bce + dice