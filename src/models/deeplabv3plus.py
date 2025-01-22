import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, backbone='resnet', in_channels=3, pretrained=True):
        super(DeepLabV3Plus, self).__init__()

        self.backbone_name = backbone

        if self.backbone_name == 'resnet':
            self.backbone = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
            input_features = 2048  # Feature size from ResNet backbone
        elif self.backbone_name == 'mobilenet':
            self.backbone = torchvision.models.segmentation.deeplabv3_mobilenet_v2(pretrained=pretrained)
            input_features = 1280  # Feature size from MobileNet backbone
        else:
            raise ValueError("Unsupported backbone")

        # Modify the first layer if in_channels != 3
        if in_channels != 3:
            self.backbone.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.backbone.backbone.conv1.out_channels,
                kernel_size=self.backbone.backbone.conv1.kernel_size,
                stride=self.backbone.backbone.conv1.stride,
                padding=self.backbone.backbone.conv1.padding,
                bias=False,
            )

        # Replace the classifier with a custom head
        self.backbone.classifier = DeepLabHead(input_features, num_classes)

    def forward(self, x):
        return self.backbone(x)["out"]