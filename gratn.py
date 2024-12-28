import torch
import torch.nn as nn
import torch.nn.functional as F


class GrATN(nn.Module):
    def __init__(self, beta=1.5, innerlayer=256, width=32, channel=3, use_batch_norm=False):
        super().__init__()
        self.beta = beta
        self.channel = channel
        self.width = width
        self.use_batch_norm = use_batch_norm

        # Convolutional layers for image and gradient processing (with optional batch normalization)
        conv_layers = [
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]

        if use_batch_norm:
            conv_layers.insert(2, nn.BatchNorm2d(16)) 
            conv_layers.insert(6, nn.BatchNorm2d(32))

        self.layer1_conv = nn.Sequential(*conv_layers)
        self.layer2_conv = nn.Sequential(*conv_layers)  

        # Attention Module
        self.attn = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Sigmoid()
        )

        # Fully connected layers for perturbation generation
        self.layer3 = nn.Sequential(
            nn.Linear(32 * 8 * 8 * 2, innerlayer),
            nn.ReLU()
        )
        
        # Increased the size of layer4 input to accommodate gradients
        self.layer4 = nn.Sequential(
            nn.Linear(innerlayer + 3 * 8 * 8, width * width * channel) 
        )

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

        self.out_image = nn.Sigmoid()
#         self.out_image = nn.Tanh()

    def forward(self, x_image, x_grad):
        self.batch_size = x_image.size(0)

        x1 = self.layer1_conv(x_image)  # Output shape: (batch_size, 32, 8, 8)
        x2 = self.layer2_conv(x_grad)   # Output shape: (batch_size, 32, 8, 8)

        # Apply attention to x1 and x2
        attn_weights = self.attn(x1)     
        x1_attn = x1 * attn_weights      
        attn_weights = self.attn(x2)    
        x2_attn = x2 * attn_weights      

        # Flatten the outputs of the convolutional layers
        x1_attn = x1_attn.view(self.batch_size, -1)  # (batch_size, 32*8*8)
        x2_attn = x2_attn.view(self.batch_size, -1)  # (batch_size, 32*8*8)

        x = torch.cat((x1_attn, x2_attn), dim=1) 
        
        # Resize x_grad to match x's spatial dimensions (32x8x8) before adding
        x_grad_resized = F.interpolate(x_grad, size=(8, 8), mode='bilinear', align_corners=False)
        # Flatten the resized gradients
        x_grad_resized = x_grad_resized.view(self.batch_size, -1)

        x = self.layer3(x)

        # Concatenate features and gradients before layer4
        x = torch.cat([x, x_grad_resized], dim=1)  

        x = self.layer4(x)
        
        x = F.interpolate(x.view(x.size(0), self.channel, self.width, self.width), size=(32, 32), mode='bilinear', align_corners=False).view(x_image.size(0), -1)

        x = self.out_image((x + x_image.view(x_image.size(0), -1) - 0.5) * 5)

        return x.view(x.size(0), self.channel, self.width, self.width)
