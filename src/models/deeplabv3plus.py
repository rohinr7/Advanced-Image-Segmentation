import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, backbone='resnet', pretrained=True):
        super(DeepLabV3Plus, self).__init__()

        # Load the pre-trained DeepLabV3 model from torchvision
        self.backbone_name = backbone
        
        if self.backbone_name == 'resnet':
            self.backbone = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        elif self.backbone_name == 'mobilenet':
            self.backbone = torchvision.models.segmentation.deeplabv3_mobilenet_v2(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone")

        # Replace the classifier head with a new one (for custom number of classes)
        self.backbone.classifier = DeepLabHead(2048, num_classes)
        
        # Decoder (Upsampling the output of the backbone)
        self.decoder = self._build_decoder(num_classes)
        
    def _build_decoder(self, num_classes):
        """ Build the decoder part of the DeepLabV3+ architecture """
        decoder = nn.Sequential(
            # Upsample the feature map to the input size
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=6, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=12, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=18, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        )
        return decoder
    
    def forward(self, x):
        # Pass through the backbone for feature extraction
        features = self.backbone.backbone(x)['out']
        
        # Pass through the decoder for refinement and segmentation map
        output = self.decoder(features)
        
        return output