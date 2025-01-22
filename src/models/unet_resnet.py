import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UNetWithResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=34, pretrained=True, resnet_variant='resnet50', use_dropout=True):
        super(UNetWithResNet, self).__init__()

        self.use_dropout = use_dropout

        # Load the specified ResNet model as the encoder backbone
        self.encoder = self._get_resnet_backbone(resnet_variant, pretrained)

        # Adjust the first layer of the encoder to match the input channels
        if in_channels != 3:
            self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Contracting path (encoder)
        self.enc1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.enc2 = nn.Sequential(self.encoder.layer1)
        self.enc3 = nn.Sequential(self.encoder.layer2)
        self.enc4 = nn.Sequential(self.encoder.layer3)
        self.enc5 = nn.Sequential(self.encoder.layer4)

        # Bottleneck with dilated convolutions
        self.bottleneck = self.conv_block(2048, 1024, dilation=2)

        # Expanding path (decoder)
        self.upconv5 = self.upconv(1024, 512)
        self.dec5 = self.conv_block(2048 + 512, 512)  # Adjusted for concatenation
        self.upconv4 = self.upconv(512, 256)
        self.dec4 = self.conv_block(1024 + 256, 256)
        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(512 + 128, 128)
        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(256 + 64, 64)

        # Final output block
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _get_resnet_backbone(self, variant, pretrained):
        """Returns the specified ResNet model backbone."""
        if variant == 'resnet18':
            return models.resnet18(pretrained=pretrained)
        elif variant == 'resnet34':
            return models.resnet34(pretrained=pretrained)
        elif variant == 'resnet50':
            return models.resnet50(pretrained=pretrained)
        elif variant == 'resnet101':
            return models.resnet101(pretrained=pretrained)
        elif variant == 'resnet152':
            return models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

    def conv_block(self, in_channels, out_channels, dilation=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if self.use_dropout:
            layers.append(nn.Dropout2d(0.5))
        return nn.Sequential(*layers)

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc5 = self.enc5(F.max_pool2d(enc4, 2))

        # Bottleneck
        bottleneck = self.bottleneck(enc5)


        # Decoder
        dec5 = self.upconv5(bottleneck)
        dec5 = self.dec5(torch.cat((dec5, F.interpolate(enc5, size=dec5.shape[2:], mode='bilinear', align_corners=False)), dim=1))
        dec4 = self.upconv4(dec5)
        dec4 = self.dec4(torch.cat((dec4, F.interpolate(enc4, size=dec4.shape[2:], mode='bilinear', align_corners=False)), dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((dec3, F.interpolate(enc3, size=dec3.shape[2:], mode='bilinear', align_corners=False)), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((dec2, F.interpolate(enc2, size=dec2.shape[2:], mode='bilinear', align_corners=False)), dim=1))


        # Final upsampling to match input resolution
        out = self.out_conv(dec2)
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        return torch.softmax(out, dim=1)
